"""
Shared test fixtures using real models.

Models are loaded once per session to amortize the ~2-3s load time.

Naming convention:
  sender_model_and_tok / receiver_model_and_tok — original Qwen→Gemma pipeline
  gemma_model_and_tok / qwen_model_and_tok — model-specific (used by both pipelines)
"""

import mlx.core as mx
import mlx_lm
import pytest

from mlxmas.latent_comm import compute_alignment


@pytest.fixture(scope="session")
def qwen_model_and_tok():
    """Load real Qwen3-4B-8bit model (once per session)."""
    model, tok = mlx_lm.load("mlx-community/Qwen3-4B-8bit")
    return model, tok


@pytest.fixture(scope="session")
def gemma_model_and_tok():
    """Load real Gemma-2-2B-it-8bit model (once per session)."""
    model, tok = mlx_lm.load("mlx-community/gemma-2-2b-it-8bit")
    return model, tok


# Legacy aliases for existing tests (Qwen→Gemma pipeline)
@pytest.fixture(scope="session")
def sender_model_and_tok(qwen_model_and_tok):
    """Alias: Qwen as sender."""
    return qwen_model_and_tok


@pytest.fixture(scope="session")
def receiver_model_and_tok(gemma_model_and_tok):
    """Alias: Gemma as receiver."""
    return gemma_model_and_tok


@pytest.fixture(scope="session")
def sender_alignment(qwen_model_and_tok):
    """Compute sender self-alignment (W_a, target_norm) once per session."""
    model, _ = qwen_model_and_tok
    W_a, target_norm = compute_alignment(model)
    return W_a, target_norm
