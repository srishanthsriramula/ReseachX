---
tags: [theorem, jacobian, stratified-hierarchy, proof]
backlinks: "[[00_Index/00_Index_MOC|Index]], [[01_Generations/v11_42Run_Confirmation_Stratified_Hierarchy|v11]]"
---

# 📐 Theorem 2: Jacobian Condition Number Explosion in Bottlenecks

### Statement
Editing $K$ contiguous transformer layers ($L \in [l_1, l_1 + K]$) causes the output Jacobian condition number to compound exponentially with depth:
$$\kappa(J_{l_1 \to l_1 + K}) \sim \prod_{l=l_1}^{l_1 + K} \|W_l + \Delta W_l\| \approx e^{K \sigma_{\max}}$$
In contrast, distributing edits across stratified early-to-mid spans separated by $\Delta l$ unedited contractive layers keeps condition number growth linear:
$$\kappa(J_{\text{stratified}}) \sim 1 + K \sigma_{\max} \rho^{\Delta l} \quad (\rho < 1)$$

### Consequence
Stratified layer placement (`[1, 2, 8, 11, 12, 16, 21, 26]`) avoids bottleneck congestion and maintains numerical representation stability. $\blacksquare$
