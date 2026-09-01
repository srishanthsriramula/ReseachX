"""v25: constrained LoRA experiment for surgical fine-tuning.

This is intentionally not a claim of mathematical invariance.  It tests a
behavioural claim: target reasoning can improve while a held-out code suite
stays within a predeclared retention tolerance.

The companion notebook is the intended entry point.  This module keeps the
protocol implementation versionable and makes it easier to audit than a large
notebook cell with mutable state.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import io
import json
import math
import os
import random
import re
import time
import urllib.request
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer


# Final evaluation data must never be used for basis construction, training,
# optimizer/lambda selection, or prompt-level debugging.
DEFAULT_CONFIG: dict[str, Any] = {
    "model_id": "poolside/Laguna-XS.2",
    "workdir": Path.cwd(),
    "artifact_dirname": "v25_artifacts",
    "global_seed": 20260901,
    "seeds": [107, 211, 503, 719, 941],
    "layers": [1, 2, 4, 8, 11, 12, 14, 16, 18, 21, 24, 26],
    # This fixed q/v budget is shared by every arm.  Do not tune it using the
    # final GPQA or final code sets.
    "target_modules": ["q_proj", "v_proj"],
    "lora_rank": 63,
    "lora_alpha": 63,
    "train_steps": 32,
    "train_batch": 4,
    "train_lr": 1.2e-5,
    "lr_min": 2e-6,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "train_seq_len": 384,
    "eval_prompt_len": 768,
    "eval_tokens": 1024,
    "eval_batch": 12,
    "retention_lambda": 1.0,
    "retention_temperature": 1.0,
    "control_nll_tolerance": 0.02,
    "target_calibration_fraction": 0.80,
    "code_retain_train": 100,
    "code_development": 32,
    "covariance_cases": 128,
    "covariance_max_tokens": 256,
    "covariance_stride": 4,
    "covariance_ridge": 0.05,
    "bootstrap_draws": 20_000,
    "run_activation_whitened_arm": True,
}


def make_config(**overrides: Any) -> dict[str, Any]:
    """Return a copy of the preregistration-style v25 configuration."""
    config = dict(DEFAULT_CONFIG)
    config.update(overrides)
    config["workdir"] = Path(config["workdir"]).resolve()
    config["artifact_dir"] = config["workdir"] / config["artifact_dirname"]
    config["results_dir"] = config["artifact_dir"] / "results"
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(destination)


def atomic_json(value: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, default=str)
    temporary.replace(destination)


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("v25 is configured for the CUDA/ROCm GPU environment used by prior Laguna runs.")
    return torch.device("cuda")


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def fetch_gpqa_csv(variant: str, token: str | None) -> list[dict[str, str]]:
    url = f"https://huggingface.co/datasets/Idavidrein/gpqa/resolve/main/{variant}.csv"
    headers = {"User-Agent": "ResearchX-v25"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return list(csv.DictReader(io.StringIO(response.read().decode("utf-8"))))


def make_multiple_choice_prompt(question: str, choices: list[str]) -> str:
    return (
        f"Question: {question}\n\nChoices:\n"
        f"(A) {choices[0]}\n(B) {choices[1]}\n(C) {choices[2]}\n(D) {choices[3]}\n\n"
        "Derive the answer step by step, then state the final answer letter in \\boxed{}."
    )


def choice_row(raw: dict[str, str], example_id: str, split: str, shuffle_seed: int) -> dict[str, str]:
    question = raw["Question"].strip()
    correct_answer = raw["Correct Answer"].strip()
    choices = [
        correct_answer,
        raw["Incorrect Answer 1"].strip(),
        raw["Incorrect Answer 2"].strip(),
        raw["Incorrect Answer 3"].strip(),
    ]
    random.Random(shuffle_seed).shuffle(choices)
    answer_letter = "ABCD"[choices.index(correct_answer)]
    return {
        "example_id": example_id,
        "split": split,
        "prompt": make_multiple_choice_prompt(question, choices),
        "reference": f"The correct answer is {correct_answer}. \\boxed{{{answer_letter}}}",
        "target_answer": answer_letter,
        "question_sha256": sha256_text(question),
    }


def build_datasets(config: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    """Build strict train/dev/final splits and write their immutable manifest."""
    token = hf_token()
    seed_everything(config["global_seed"])

    diamond = fetch_gpqa_csv("gpqa_diamond", token)
    gpqa_final = [choice_row(row, f"gpqa_diamond_{i:04d}", "final", 200_000 + i) for i, row in enumerate(diamond)]
    final_questions = {row["question_sha256"] for row in gpqa_final}

    main = fetch_gpqa_csv("gpqa_main", token)
    main_rows = [choice_row(row, f"gpqa_main_{i:04d}", "candidate", 300_000 + i) for i, row in enumerate(main)]
    # Exact full-question checks, not the old first-100-character heuristic.
    main_rows = [row for row in main_rows if row["question_sha256"] not in final_questions]
    random.Random(config["global_seed"] + 17).shuffle(main_rows)
    cut = int(len(main_rows) * config["target_calibration_fraction"])
    gpqa_train, gpqa_dev = main_rows[:cut], main_rows[cut:]

    sciq = load_dataset("allenai/sciq", split="train", trust_remote_code=True)
    sciq_train: list[dict[str, str]] = []
    for i, item in enumerate(sciq):
        if len(sciq_train) >= 1_000:
            break
        distractors = [item.get("distractor1", ""), item.get("distractor2", ""), item.get("distractor3", "")]
        if not all(distractors):
            continue
        correct = item["correct_answer"]
        choices = [correct] + distractors
        random.Random(400_000 + i).shuffle(choices)
        letter = "ABCD"[choices.index(correct)]
        explanation = item.get("support", "") or f"The answer is {correct}."
        sciq_train.append(
            {
                "example_id": f"sciq_{i:04d}",
                "split": "train",
                "prompt": make_multiple_choice_prompt(item["question"], choices),
                "reference": f"{explanation}\n\\boxed{{{letter}}}",
                "target_answer": letter,
                "question_sha256": sha256_text(item["question"]),
            }
        )

    humaneval = load_dataset("openai/openai_humaneval", split="test", trust_remote_code=True)
    code_rows = [
        {
            "example_id": str(item["task_id"]),
            "prompt": f"Complete this code:\n{item['prompt']}",
            "reference": item["canonical_solution"],
        }
        for item in humaneval
        if item.get("prompt") and item.get("canonical_solution")
    ]
    random.Random(config["global_seed"] + 31).shuffle(code_rows)
    n_retain = config["code_retain_train"]
    n_dev = config["code_development"]
    if len(code_rows) <= n_retain + n_dev:
        raise ValueError("HumanEval split leaves no untouched final control tasks.")
    code_retain_train = code_rows[:n_retain]
    code_dev = code_rows[n_retain : n_retain + n_dev]
    code_final = code_rows[n_retain + n_dev :]

    result = {
        "target_train": gpqa_train + sciq_train,
        "target_dev": gpqa_dev,
        "target_final": gpqa_final,
        "code_retain_train": code_retain_train,
        "code_dev": code_dev,
        "code_final": code_final,
    }
    assert not ({row["question_sha256"] for row in gpqa_train} & final_questions)
    manifest = {
        "protocol": "v25_surgical_constrained_lora",
        "final_data_rule": "No target_final or code_final item may be used for basis construction, training, or selection.",
        "counts": {name: len(rows) for name, rows in result.items()},
        "ids": {name: [row["example_id"] for row in rows] for name, rows in result.items()},
        "config": {key: value for key, value in config.items() if key not in {"workdir", "artifact_dir", "results_dir"}},
    }
    atomic_json(manifest, config["artifact_dir"] / "split_manifest.json")
    for name, rows in result.items():
        atomic_csv(pd.DataFrame(rows), config["artifact_dir"] / f"{name}.csv")
    return result


def resolve_model(model_id: str) -> str:
    candidates = [
        Path("/shared-docker/models/Laguna-XS.2"),
        Path("/shared-docker/Laguna-XS.2"),
        Path("/workspace/models/Laguna-XS.2"),
        Path.home() / "models" / "Laguna-XS.2",
    ]
    for candidate in candidates:
        if (candidate / "config.json").exists():
            return str(candidate)
    return model_id


def load_laguna(config: dict[str, Any], device: torch.device) -> tuple[Any, Any]:
    """Load Laguna and fuse MoE weights when the custom loader leaves them compressed."""
    token = hf_token()
    model_path = resolve_model(config["model_id"])
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Loading {model_path} …", flush=True)
    model, loading_info = AutoModelForCausalLM.from_pretrained(
        model_path,
        token=token,
        trust_remote_code=True,
        device_map={"": 0},
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        attn_implementation="eager",
        output_loading_info=True,
    )
    model.eval()
    model.config.use_cache = False

    shards = sorted(Path(model_path).glob("**/*.safetensors")) if Path(model_path).exists() else []
    shards = [path for path in shards if path.stat().st_size > 100 * 1024 * 1024]
    fused = 0
    for shard in shards:
        try:
            state = load_file(str(shard), device="cpu")
        except Exception:
            continue
        with torch.no_grad():
            for layer_index, layer in enumerate(model.model.layers):
                mlp = getattr(layer, "mlp", None)
                if not (mlp and hasattr(mlp, "experts") and hasattr(mlp.experts, "down_proj")):
                    continue
                for expert_index in range(256):
                    down_key = f"model.layers.{layer_index}.mlp.experts.{expert_index}.down_proj.weight"
                    gate_key = f"model.layers.{layer_index}.mlp.experts.{expert_index}.gate_proj.weight"
                    up_key = f"model.layers.{layer_index}.mlp.experts.{expert_index}.up_proj.weight"
                    if down_key in state:
                        target = mlp.experts.down_proj
                        target[expert_index].copy_(state[down_key].to(device=target.device, dtype=target.dtype))
                        fused += 1
                    if gate_key in state and up_key in state:
                        target = mlp.experts.gate_up_proj
                        target[expert_index].copy_(torch.cat([state[gate_key], state[up_key]], dim=0).to(device=target.device, dtype=target.dtype))
        del state
        gc.collect()
    if fused:
        print(f"Fused {fused} expert tensors.", flush=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    del loading_info
    gc.collect()
    torch.cuda.empty_cache()
    return model, tokenizer


def chat_prefix_text(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_case(tokenizer: Any, prompt: str, reference: str) -> tuple[list[int], int]:
    prefix = chat_prefix_text(tokenizer, prompt)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    full_ids = tokenizer.encode(prefix + "\n" + reference, add_special_tokens=False)
    start = 0
    for prefix_token, full_token in zip(prefix_ids, full_ids):
        if prefix_token != full_token:
            break
        start += 1
    if start <= 0 or start >= len(full_ids):
        full_ids = prefix_ids + tokenizer.encode("\n" + reference, add_special_tokens=False)
        start = len(prefix_ids)
    return full_ids, start


def tokenize_supervised(tokenizer: Any, rows: list[dict[str, str]], config: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    encoded: list[tuple[list[int], int]] = []
    for row in rows:
        ids, start = parse_case(tokenizer, row["prompt"], row["reference"])
        ids = ids[: config["train_seq_len"]]
        start = min(start, len(ids) - 1)
        encoded.append((ids, start))
    max_length = max(len(ids) for ids, _ in encoded)
    pad = tokenizer.pad_token_id or 0
    input_ids = torch.full((len(encoded), max_length), pad, dtype=torch.long)
    attention = torch.zeros((len(encoded), max_length), dtype=torch.long)
    labels = torch.full((len(encoded), max_length), -100, dtype=torch.long)
    for i, (ids, start) in enumerate(encoded):
        length = len(ids)
        input_ids[i, :length] = torch.tensor(ids)
        attention[i, :length] = 1
        labels[i, start:length] = torch.tensor(ids[start:length])
    return {"input_ids": input_ids.to(device), "attention_mask": attention.to(device), "labels": labels.to(device)}


def sampled_batch(pool: dict[str, torch.Tensor], batch_size: int, rng: np.random.Generator) -> dict[str, torch.Tensor]:
    n = pool["input_ids"].shape[0]
    indices = torch.tensor(rng.integers(0, n, size=batch_size), device=pool["input_ids"].device)
    return {key: value[indices] for key, value in pool.items()}


def find_target_modules(model: Any, requested: Iterable[str], layers: Iterable[int]) -> list[str]:
    requested = set(requested)
    layers = set(layers)
    found: set[str] = set()
    for name, module in model.named_modules():
        match = re.search(r"layers\.(\d+)\.", name)
        if match and int(match.group(1)) in layers and isinstance(module, nn.Linear):
            suffix = name.rsplit(".", 1)[-1]
            if suffix in requested:
                found.add(suffix)
    missing = requested - found
    if missing:
        raise RuntimeError(f"Requested LoRA modules were not found: {sorted(missing)}")
    return sorted(found)


def collect_activation_covariances(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    config: dict[str, Any],
) -> dict[tuple[int, str], torch.Tensor]:
    """Activation covariance for the explicitly exploratory initializer only.

    It is not called Fisher information and is never constructed from final
    target/control items.  Statistics are accumulated as sufficient moments
    to avoid retaining every activation tensor.
    """
    layers = set(config["layers"])
    target_modules = set(config["target_modules"])
    stats: dict[tuple[int, str], dict[str, torch.Tensor | int]] = {}
    hooks = []

    def hook_for(key: tuple[int, str]):
        def capture(_module: Any, inputs: tuple[torch.Tensor, ...], _output: Any) -> None:
            if not inputs or inputs[0].dim() != 3:
                return
            x = inputs[0][0, :: config["covariance_stride"], :].detach().float().cpu()
            if not len(x):
                return
            if key not in stats:
                stats[key] = {"sum": torch.zeros(x.shape[1]), "cross": torch.zeros((x.shape[1], x.shape[1])), "n": 0}
            stats[key]["sum"] += x.sum(dim=0)
            stats[key]["cross"] += x.T @ x
            stats[key]["n"] += x.shape[0]
        return capture

    for name, module in model.named_modules():
        layer_match = re.search(r"layers\.(\d+)\.", name)
        if not layer_match or not isinstance(module, nn.Linear):
            continue
        layer = int(layer_match.group(1))
        suffix = name.rsplit(".", 1)[-1]
        if layer in layers and suffix in target_modules:
            hooks.append(module.register_forward_hook(hook_for((layer, suffix))))

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for prompt in prompts[: config["covariance_cases"]]:
            encoded = tokenizer(
                chat_prefix_text(tokenizer, prompt),
                return_tensors="pt",
                truncation=True,
                max_length=config["covariance_max_tokens"],
            ).to(device)
            model(**encoded, use_cache=False)
    for hook in hooks:
        hook.remove()
    covariances: dict[tuple[int, str], torch.Tensor] = {}
    for key, state in stats.items():
        n = int(state["n"])
        if n < 2:
            continue
        mean = state["sum"] / n
        covariances[key] = (state["cross"] - n * torch.outer(mean, mean)) / (n - 1)
    torch.cuda.empty_cache()
    return covariances


def build_activation_whitened_bases(
    control_cov: dict[tuple[int, str], torch.Tensor],
    target_cov: dict[tuple[int, str], torch.Tensor],
    config: dict[str, Any],
) -> dict[tuple[int, str], torch.Tensor]:
    """Return a surrogate activation-energy basis, not an invariance guarantee."""
    bases: dict[tuple[int, str], torch.Tensor] = {}
    for key in sorted(set(control_cov) & set(target_cov)):
        control = control_cov[key].cuda(dtype=torch.float32)
        target = target_cov[key].cuda(dtype=torch.float32)
        ridge = config["covariance_ridge"] * torch.eye(control.shape[0], device=control.device)
        values, vectors = torch.linalg.eigh(control + ridge)
        inverse_sqrt = vectors * values.clamp_min(1e-8).rsqrt().unsqueeze(0) @ vectors.T
        whitened_target = inverse_sqrt @ target @ inverse_sqrt
        _, target_vectors = torch.linalg.eigh(whitened_target)
        basis = (target_vectors[:, -config["lora_rank"] :].T @ inverse_sqrt).cpu()
        bases[key] = basis.float()
        del control, target, ridge, values, vectors, inverse_sqrt, whitened_target, target_vectors
    torch.cuda.empty_cache()
    return bases


def layer_and_suffix(name: str) -> tuple[int, str] | None:
    match = re.search(r"layers\.(\d+)\.", name)
    if not match:
        return None
    return int(match.group(1)), name.rsplit(".", 1)[-1]


def reset_adapters(peft_model: Any, seed: int, bases: dict[tuple[int, str], torch.Tensor] | None = None) -> pd.DataFrame:
    """Reset every arm to equal capacity; whitening changes only A's direction."""
    seed_everything(seed)
    rows = []
    for name, module in peft_model.named_modules():
        if not hasattr(module, "lora_A") or "default" not in module.lora_A:
            continue
        a = module.lora_A["default"].weight
        b = module.lora_B["default"].weight
        with torch.no_grad():
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            random_norm = float(a.norm().item())
            key = layer_and_suffix(name)
            initializer = "random"
            if bases and key in bases and a.shape == bases[key].shape:
                # Match the actual random draw, not an incorrect analytical
                # approximation of a Kaiming-uniform norm.
                basis = bases[key].to(device=a.device, dtype=a.dtype)
                a.copy_(basis * (a.norm() / basis.norm().clamp_min(1e-12)))
                initializer = "activation_whitened"
            nn.init.zeros_(b)
        a.requires_grad_(True)
        b.requires_grad_(True)
        rows.append(
            {
                "module": name,
                "initializer": initializer,
                "random_A_norm": random_norm,
                "actual_A_norm": float(a.norm().item()),
                "B_norm": float(b.norm().item()),
                "A_trainable": bool(a.requires_grad),
                "B_trainable": bool(b.requires_grad),
            }
        )
    if not rows:
        raise RuntimeError("No LoRA adapters were reset; inspect target module discovery.")
    return pd.DataFrame(rows)


def continuation_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )


def retained_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    mask = labels[:, 1:].ne(-100)
    student_log_probs = F.log_softmax(student_logits[:, :-1, :].float() / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits[:, :-1, :].float() / temperature, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    per_token = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    return (per_token * mask).sum() / mask.sum().clamp_min(1) * (temperature**2)


def train_arm(
    peft_model: Any,
    target_pool: dict[str, torch.Tensor],
    retention_pool: dict[str, torch.Tensor],
    config: dict[str, Any],
    seed: int,
    retention_lambda: float,
) -> pd.DataFrame:
    """Optimise target CE plus a direct base-behaviour KL retention penalty."""
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    parameters = [parameter for parameter in peft_model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config["train_lr"],
        betas=(0.9, 0.95),
        weight_decay=config["weight_decay"],
    )
    history = []
    peft_model.train()
    for step in range(config["train_steps"]):
        target = sampled_batch(target_pool, config["train_batch"], rng)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            target_out = peft_model(input_ids=target["input_ids"], attention_mask=target["attention_mask"], use_cache=False)
            target_loss = continuation_nll(target_out.logits, target["labels"])
            if retention_lambda:
                retention = sampled_batch(retention_pool, config["train_batch"], rng)
                # The teacher is the immutable base model evaluated on exactly
                # the same retained continuation contexts. No final-control
                # item is in this pool.
                with torch.no_grad():
                    with peft_model.disable_adapter():
                        teacher_out = peft_model(
                            input_ids=retention["input_ids"],
                            attention_mask=retention["attention_mask"],
                            use_cache=False,
                        )
                retention_out = peft_model(
                    input_ids=retention["input_ids"],
                    attention_mask=retention["attention_mask"],
                    use_cache=False,
                )
                kl_loss = retained_kl(
                    retention_out.logits,
                    teacher_out.logits,
                    retention["labels"],
                    config["retention_temperature"],
                )
            else:
                kl_loss = torch.zeros((), device=target_loss.device, dtype=target_loss.dtype)
            loss = target_loss + retention_lambda * kl_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config["grad_clip"])
        progress = step / max(1, config["train_steps"] - 1)
        lr = config["lr_min"] + 0.5 * (config["train_lr"] - config["lr_min"]) * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        history.append(
            {
                "step": step + 1,
                "target_ce": float(target_loss.detach()),
                "retention_kl": float(kl_loss.detach()),
                "objective": float(loss.detach()),
                "lr": lr,
                "grad_norm_preclip": float(grad_norm),
            }
        )
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return pd.DataFrame(history)


def extract_answer(text: str) -> str:
    text = (text or "").strip()
    boxed = re.findall(r"\\boxed\{\s*([A-Da-d])\s*\}", text)
    if boxed:
        return boxed[-1].upper()
    phrase = re.findall(r"(?:the\s+answer\s+is|answer\s*:)\s*\(?([A-Da-d])\)?", text, flags=re.I)
    if phrase:
        return phrase[-1].upper()
    letters = re.findall(r"\(([A-Da-d])\)", text[-160:])
    return letters[-1].upper() if letters else ""


@torch.inference_mode()
def evaluate_gpqa(peft_model: Any, tokenizer: Any, rows: list[dict[str, str]], config: dict[str, Any], tag: str) -> pd.DataFrame:
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    records = []
    try:
        peft_model.eval()
        for start in range(0, len(rows), config["eval_batch"]):
            batch_rows = rows[start : start + config["eval_batch"]]
            prompts = [chat_prefix_text(tokenizer, row["prompt"]) for row in batch_rows]
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config["eval_prompt_len"],
            ).to(next(peft_model.parameters()).device)
            generated = peft_model.generate(
                **encoded,
                max_new_tokens=config["eval_tokens"],
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            responses = tokenizer.batch_decode(generated[:, encoded["input_ids"].shape[1] :], skip_special_tokens=True)
            for row, response in zip(batch_rows, responses):
                extracted = extract_answer(response)
                records.append(
                    {
                        "method": tag,
                        "example_id": row["example_id"],
                        "target": row["target_answer"],
                        "extracted": extracted,
                        "correct": float(extracted == row["target_answer"]),
                        # Store full response for extractor audits, not a short preview.
                        "response": response,
                    }
                )
            del generated, encoded
            torch.cuda.empty_cache()
    finally:
        tokenizer.padding_side = old_padding_side
    return pd.DataFrame(records)


@torch.inference_mode()
def evaluate_code_nll(peft_model: Any, tokenizer: Any, rows: list[dict[str, str]], config: dict[str, Any], tag: str, disable_adapter: bool = False) -> pd.DataFrame:
    records = []
    context = peft_model.disable_adapter() if disable_adapter else nullcontext()
    peft_model.eval()
    with context:
        for row in rows:
            ids, start = parse_case(tokenizer, row["prompt"], row["reference"])
            input_ids = torch.tensor(ids, device=next(peft_model.parameters()).device).unsqueeze(0)
            labels = torch.full_like(input_ids, -100)
            labels[:, start:] = input_ids[:, start:]
            output = peft_model(input_ids=input_ids, use_cache=False)
            tokens = int(labels[:, 1:].ne(-100).sum())
            nll_sum = F.cross_entropy(
                output.logits[:, :-1, :].float().reshape(-1, output.logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            records.append(
                {
                    "method": tag,
                    "example_id": row["example_id"],
                    "nll_sum": float(nll_sum),
                    "tokens": tokens,
                    "nll": float(nll_sum) / max(tokens, 1),
                }
            )
            del input_ids, labels, output
    return pd.DataFrame(records)


def hierarchical_bootstrap(delta: np.ndarray, draws: int, seed: int) -> tuple[float, float, float]:
    """Bootstrap jointly over seeds and held-out examples."""
    if delta.ndim != 2:
        raise ValueError("Expected [seed, example] delta matrix.")
    generator = np.random.default_rng(seed)
    n_seeds, n_examples = delta.shape
    samples = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        seed_indices = generator.integers(0, n_seeds, size=n_seeds)
        item_indices = generator.integers(0, n_examples, size=n_examples)
        samples[draw] = delta[np.ix_(seed_indices, item_indices)].mean()
    return float(delta.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def summarize_results(
    base_gpqa: pd.DataFrame,
    base_code: pd.DataFrame,
    arm_results: list[dict[str, Any]],
    config: dict[str, Any],
) -> pd.DataFrame:
    base_gpqa_map = base_gpqa.set_index("example_id")["correct"]
    base_code_map = base_code.set_index("example_id")["nll"]
    rows = []
    for arm in sorted({run["arm"] for run in arm_results}):
        runs = [run for run in arm_results if run["arm"] == arm]
        gpqa_delta = []
        code_delta = []
        for run in runs:
            gpqa = run["gpqa"].set_index("example_id")["correct"].reindex(base_gpqa_map.index)
            code = run["code"].set_index("example_id")["nll"].reindex(base_code_map.index)
            gpqa_delta.append((gpqa - base_gpqa_map).to_numpy())
            code_delta.append((code - base_code_map).to_numpy())
        gain, gain_low, gain_high = hierarchical_bootstrap(np.stack(gpqa_delta), config["bootstrap_draws"], config["global_seed"])
        nll_delta, nll_low, nll_high = hierarchical_bootstrap(np.stack(code_delta), config["bootstrap_draws"], config["global_seed"] + 1)
        rows.append(
            {
                "arm": arm,
                "seeds": len(runs),
                "target_gain": gain,
                "target_gain_ci_low": gain_low,
                "target_gain_ci_high": gain_high,
                "control_nll_delta": nll_delta,
                "control_nll_delta_ci_low": nll_low,
                "control_nll_delta_ci_high": nll_high,
                "target_success": gain_low > 0,
                # Positive NLL change is degradation.  This is a predeclared
                # non-inferiority threshold, not an absolute-change proxy.
                "control_retained": nll_high <= config["control_nll_tolerance"],
            }
        )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["surgical_success"] = summary["target_success"] & summary["control_retained"]
    return summary


def run_v25(config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Run the frozen v25 protocol.  Returns the final arm-level summary."""
    config = make_config(**(config or {}))
    config["artifact_dir"].mkdir(parents=True, exist_ok=True)
    config["results_dir"].mkdir(parents=True, exist_ok=True)
    atomic_json(config, config["artifact_dir"] / "protocol_config.json")
    device = require_cuda()
    data = build_datasets(config)
    model, tokenizer = load_laguna(config, device)
    target_modules = find_target_modules(model, config["target_modules"], config["layers"])
    config["target_modules"] = target_modules
    atomic_json({"target_modules": target_modules, "layers": config["layers"]}, config["artifact_dir"] / "model_targeting.json")

    # All covariance inputs are calibration-only.  The final GPQA and final
    # HumanEval partitions have not been touched at this point.
    target_cov = collect_activation_covariances(model, tokenizer, [row["prompt"] for row in data["target_train"]], config)
    control_cov = collect_activation_covariances(model, tokenizer, [row["prompt"] for row in data["code_retain_train"]], config)
    bases = build_activation_whitened_bases(control_cov, target_cov, config)
    atomic_json({"basis_count": len(bases), "note": "Activation-whitened exploratory initializer; not Fisher or a safety guarantee."}, config["artifact_dir"] / "initializer_manifest.json")

    target_pool = tokenize_supervised(tokenizer, data["target_train"], config, device)
    retention_pool = tokenize_supervised(tokenizer, data["code_retain_train"], config, device)
    lora_config = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        target_modules=target_modules,
        layers_to_transform=config["layers"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.gradient_checkpointing_enable()

    # The base result is generated once before any final-model result is read.
    reset_adapters(peft_model, config["global_seed"])
    base_gpqa = evaluate_gpqa(peft_model, tokenizer, data["target_final"], config, "base")
    base_code = evaluate_code_nll(peft_model, tokenizer, data["code_final"], config, "base", disable_adapter=True)
    atomic_csv(base_gpqa, config["results_dir"] / "base_gpqa_full_responses.csv")
    atomic_csv(base_code, config["results_dir"] / "base_code_final_nll.csv")

    arms = [
        {"name": "standard_target_only", "retention_lambda": 0.0, "bases": None},
        {"name": "standard_plus_retention_kl", "retention_lambda": config["retention_lambda"], "bases": None},
    ]
    if config["run_activation_whitened_arm"]:
        arms.append(
            {
                "name": "activation_whitened_plus_retention_kl_exploratory",
                "retention_lambda": config["retention_lambda"],
                "bases": bases,
            }
        )

    all_runs: list[dict[str, Any]] = []
    for arm in arms:
        for seed in config["seeds"]:
            print(f"\n=== {arm['name']} | seed {seed} ===", flush=True)
            initial = reset_adapters(peft_model, seed, arm["bases"])
            train_history = train_arm(
                peft_model,
                target_pool,
                retention_pool,
                config,
                seed,
                arm["retention_lambda"],
            )
            tag = f"{arm['name']}_s{seed}"
            gpqa = evaluate_gpqa(peft_model, tokenizer, data["target_final"], config, tag)
            code = evaluate_code_nll(peft_model, tokenizer, data["code_final"], config, tag)
            atomic_csv(initial, config["results_dir"] / f"{tag}_initialization.csv")
            atomic_csv(train_history, config["results_dir"] / f"{tag}_train_history.csv")
            atomic_csv(gpqa, config["results_dir"] / f"{tag}_gpqa_full_responses.csv")
            atomic_csv(code, config["results_dir"] / f"{tag}_code_final_nll.csv")
            all_runs.append({"arm": arm["name"], "seed": seed, "gpqa": gpqa, "code": code})

    summary = summarize_results(base_gpqa, base_code, all_runs, config)
    atomic_csv(summary, config["results_dir"] / "v25_final_summary.csv")
    print("\n=== v25 final summary ===")
    print(summary.to_string(index=False))
    return summary
