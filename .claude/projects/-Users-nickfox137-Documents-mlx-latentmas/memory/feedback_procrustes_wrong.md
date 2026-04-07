---
name: Procrustes is wrong for cross-model
description: Procrustes alignment is fundamentally wrong for cross-model latent communication — do not use it or suggest it
type: feedback
---

Procrustes is the wrong tool for cross-model latent communication. It assumes isometry, known point correspondence, and linearity — none of which hold between independently trained models with different architectures. 0.72 cosine similarity means nothing when the receiver scores 0/5. The user has been through this multiple times and does not want to revisit Procrustes-based approaches.

**Why:** The geometric structures of Qwen and Gemma's representation spaces are fundamentally different. No rotation matrix can align them in a way that preserves functional meaning. Same-model works because the geometry is identical — cross-model needs a completely different approach.

**How to apply:** Never suggest Procrustes variants, layer pair sweeps, or alignment quality improvements as a path forward for cross-model communication. Focus on approaches that don't assume isometric spaces — direct KV projection, contrastive learning, or nonlinear adapters.
