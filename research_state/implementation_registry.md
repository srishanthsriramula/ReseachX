# IMPLEMENTATION REGISTRY

Registry of mathematical, algorithmic, and software implementations developed across the project.

---

### 1. Direct BF16 Model Loading & Weight Mapping
- **Purpose**: Direct instantiation of Laguna XS.2 without dynamic runtime decompression.
- **Key Modules**:
  - `register_laguna_conversion_mapping()`: Maps safetensors shard keys to native PyTorch `nn.Module` weights.
  - `load_bf16_model()`: Direct GPU allocation occupying 62.29 GiB VRAM on RTX PRO 6000.

---

### 2. Fixed-Routing Causal Ablation Hooks
- **Purpose**: Measure counterfactual necessity of individual experts while freezing router decisions.
- **Implementation**:
  ```python
  def causal_ablation_hook(module, inputs, outputs, expert_idx):
      # outputs shape: [batch, tokens, num_selected, hidden_dim]
      # Zero out contribution of expert_idx while preserving gate weights
      mask = (module.selected_experts == expert_idx)
      outputs[mask] = 0.0
      return outputs
  ```

---

### 3. Exact Token-Level Cross-Entropy Scorer
- **Purpose**: Eliminates selective-logit approximations and enforces token alignment.
- **Implementation**:
  ```python
  def compute_aligned_nll(model, input_ids, target_mask):
      logits = model(input_ids).logits
      shift_logits = logits[..., :-1, :].contiguous()
      shift_labels = input_ids[..., 1:].contiguous()
      loss_fct = nn.CrossEntropyLoss(reduction='none')
      loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
      masked_loss = loss * target_mask[..., 1:].contiguous().view(-1)
      return masked_loss.sum() / target_mask[..., 1:].sum()
  ```

---

### 4. Behavioral Repair Kernel (BRK) & Null-Space Projector
- **Purpose**: Project failure-repair preference gradients orthogonally to preserved capability Fisher information.
- **Mathematical Formulation**:
  $$P_\perp = I - U_K U_K^\top, \quad U_K = \text{top-}K \text{ eigenvectors of } \sum_{j \in \text{Preserved}} g_j^{\text{pres}} (g_j^{\text{pres}})^\top$$
  $$g^{\text{projected}} = P_\perp \left( \nabla_\theta \log P_\theta(y^+ \mid x) - \nabla_\theta \log P_\theta(y^- \mid x) \right)$$
