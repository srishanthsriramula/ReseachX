---
tags: [theorem, moe, routing-bifurcation, proof]
backlinks: "[[00_Index/00_Index_MOC|Index]], [[01_Generations/v5_Matched_Bakeoff_Router_Avalanche|v5]]"
---

# 📐 Theorem 1: Discontinuous MoE Routing Bifurcation

### Statement
In a sparse Mixture-of-Experts architecture with Top-$k$ softmax gating, expert parameter matrices are mutually orthogonal ($\|W_i - W_j\|_F = \Omega(1)$). For any perturbation $\Delta W$ applied to a routed expert, the output activation shift induces discontinuous permutations in downstream routing decisions:
$$\lim_{\|\Delta W\| \to 0} \|\Delta \text{MoE}(x)\| = \Omega(1)$$

### Proof Sketch
1. Router logits are given by $z_i(x) = w_i^T x$. The Top-$k$ set is $\mathcal{E}_k(x) = \operatorname{arg\,topk}_i (z_i(x))$.
2. Modifying expert $e$ alters its output $h_e(x) = (W_e + \Delta W_e) x$.
3. The perturbation shifts downstream inputs $x_{l+1} \leftarrow x_{l+1} + \Delta h_e$.
4. Whenever $z_i(x_{l+1})$ and $z_j(x_{l+1})$ are within $\epsilon = \|w_i - w_j\| \|\Delta h_e\|$, the Top-$k$ expert set permutes.
5. Because expert weights are orthogonal, replacing expert $i$ with expert $j$ creates an $\mathcal{O}(1)$ jump in residual output:
$$\|E_i(x) - E_j(x)\| \ge \sigma_{\min}(W_i - W_j) \|x\| = \Omega(1)$$
This triggers a discrete cascade across all subsequent layers. $\blacksquare$
