import json
import os
from pathlib import Path

# Load original notebook
with open('laguna/laguna_xs2_v11_fresh_confirmation_random_placement.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update title cell
nb['cells'][0]['source'] = [
    "# Laguna XS.2 v11 — Fresh Confirmation + Random-Placement Distribution (AMD Instinct MI300X Optimized)\n",
    "\n",
    "This notebook is **confirmatory**, not exploratory, optimized for **AMD Instinct MI300X (192GB HBM3)**.\n",
    "\n",
    "### Hardware Optimization Profile: AMD Instinct MI300X\n",
    "- **192 GB HBM3 @ 5.3 TB/s**: Scales `GENERATION_BATCH_SIZE` from 4 to **32** (reducing 384 test items from 96 batches to 12 batches).\n",
    "- **High-Speed Model Download**: Uses `hf_transfer` multi-threaded transfer for rapid 62GB BF16 checkpoint resolution.\n",
    "- **ROCm / HIP Tuning**: Configures `PYTORCH_HIP_ALLOC_CONF=\"expandable_segments:True\"` and SDPA attention kernels.\n",
    "- **Zero Protocol Drift**: Preserves exact frozen seeds, 384 fresh test items, signature-matched random LoRA plans, and bootstrap statistics.\n",
    "\n",
    "### Primary A\n",
    "Confirm whether the frozen K=4 writable expert set at the v10-successful **8-update dose** improves autonomous GSM8K accuracy on a fresh test set.\n",
    "\n",
    "### Primary B\n",
    "Test whether the frozen v10 guided 8-layer LoRA placement beats a distribution of random 8-layer placements when **rank, LR, update count, training data, layer-shape signature, parameter count, and training seeds are matched**.\n"
]

# Update Cell 2 (Pip install for ROCm / MI300X)
nb['cells'][2]['source'] = [
    "%pip -q install --upgrade pip\n",
    "%pip -q install \\\n",
    "    hf_transfer \\\n",
    "    huggingface_hub \\\n",
    "    transformers==4.49.0 \\\n",
    "    peft==0.14.0 \\\n",
    "    accelerate==1.4.0 \\\n",
    "    datasets==3.3.2 \\\n",
    "    tqdm \\\n",
    "    pandas \\\n",
    "    numpy \\\n",
    "    scipy \\\n",
    "    matplotlib \\\n",
    "    seaborn \\\n",
    "    safetensors \\\n",
    "    psutil\n",
    "\n",
    "print('Dependencies installed. Note: On ROCm environments, ensure PyTorch with ROCm 6.x is active.')\n"
]

# Update Cell 3 & 4 (Runtime tuning for MI300X)
nb['cells'][3]['source'] = ["## 2 — Runtime tuning for AMD Instinct MI300X (ROCm / HIP)\n"]
nb['cells'][4]['source'] = [
    "import os\n",
    "import sys\n",
    "import gc\n",
    "import time\n",
    "import shutil\n",
    "import psutil\n",
    "from pathlib import Path\n",
    "\n",
    "# Enable high-throughput HuggingFace transfer\n",
    "os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'\n",
    "os.environ['TOKENIZERS_PARALLELISM'] = 'false'\n",
    "\n",
    "# ROCm / HIP memory and kernel tuning\n",
    "os.environ['HIP_VISIBLE_DEVICES'] = os.environ.get('HIP_VISIBLE_DEVICES', '0')\n",
    "os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'expandable_segments:True'\n",
    "os.environ['OMP_NUM_THREADS'] = str(min(16, os.cpu_count() or 16))\n",
    "\n",
    "# Workspace paths\n",
    "WORK_ROOT = Path(os.environ.get('RESEARCH_ROOT', Path.cwd())).resolve()\n",
    "print('Working root:', WORK_ROOT)\n"
]

# Update Cell 9 & 10 (Hardware preflight for MI300X)
nb['cells'][9]['source'] = ["## 5 — Hardware and storage preflight for AMD Instinct MI300X\n"]
nb['cells'][10]['source'] = [
    "import torch\n",
    "import psutil\n",
    "\n",
    "print('=== Host Environment ===')\n",
    "print(f'Python: {sys.version.split()[0]}')\n",
    "print(f'Logical CPUs: {os.cpu_count()}')\n",
    "print(f'RAM Total:     {psutil.virtual_memory().total / (1024**3):.2f} GiB')\n",
    "print(f'RAM Available: {psutil.virtual_memory().available / (1024**3):.2f} GiB')\n",
    "\n",
    "print('\\n=== GPU Accelerator (ROCm / CUDA) ===')\n",
    "print(f'Torch version:  {torch.__version__}')\n",
    "print(f'CUDA/ROCm available: {torch.cuda.is_available()}')\n",
    "if torch.cuda.is_available():\n",
    "    device_name = torch.cuda.get_device_name(0)\n",
    "    total_vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)\n",
    "    print(f'Device Name:    {device_name}')\n",
    "    print(f'Total VRAM:     {total_vram_gib:.2f} GiB')\n",
    "    if total_vram_gib < 80:\n",
    "        print('WARNING: Total VRAM is below 80 GiB; Laguna XS.2 BF16 requires at least ~65 GiB resident.')\n",
    "    else:\n",
    "        print('Hardware preflight: PASS (High-capacity accelerator detected)')\n",
    "else:\n",
    "    raise RuntimeError('No GPU accelerator detected.')\n"
]

# Update Cell 12 (Protocol constants & Batch Size Scaling for MI300X)
original_cell_12 = ''.join(nb['cells'][12]['source'])
# Replace batch sizes in cell 12
updated_cell_12 = original_cell_12.replace('EVAL_BATCH_SIZE = 8', 'EVAL_BATCH_SIZE = 32')
updated_cell_12 = updated_cell_12.replace('GENERATION_BATCH_SIZE = 4', 'GENERATION_BATCH_SIZE = 32')
nb['cells'][12]['source'] = [updated_cell_12]

# Update Cell 19 & 20 (Model download with hf_transfer and multi-workers)
nb['cells'][19]['source'] = ["## 10 — High-speed parallel download & checkpoint resolution\n"]
original_cell_20 = ''.join(nb['cells'][20]['source'])
# Replace max_workers=2 with max_workers=8 for fast MI300X cloud environments
updated_cell_20 = original_cell_20.replace('max_workers=2', 'max_workers=8')
nb['cells'][20]['source'] = [updated_cell_20]

# Update Cell 23 (Loading onto accelerator)
nb['cells'][23]['source'] = ["## 12 — Load BF16 model directly onto AMD Instinct MI300X\n"]

# Save as new optimized notebook
out_path = Path('laguna/laguna_xs2_v11_fresh_confirmation_mi300x_optimized.ipynb')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"Successfully generated {out_path} ({out_path.stat().st_size} bytes)")
