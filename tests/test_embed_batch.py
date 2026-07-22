"""
Backward-compatibility and batching tests for ``relsim.relsim_score``.

The batched ``embed`` change must not alter the existing single-image behavior.
These tests verify:

  * ``preprocess_image`` still works for a single image, and now also for a list.
  * ``embed(img)`` still returns a ``[1, 384]`` tensor (unchanged contract).
  * ``embed([...])`` returns ``[N, 384]`` from a SINGLE batched forward pass
    (not N calls), and is order-preserving / consistent with the single path.
  * ``__call__(img1, img2)`` returns the same scalar similarity as before, now
    via one batched forward.

Most tests use a lightweight fake processor/model so they run on CPU without
downloading the 7B checkpoint. The final test does a real single-vs-batch
numerical parity check; it runs on CPU or CUDA and is skipped only when no
checkpoint is available (set RELSIM_CHECKPOINT_DIR to enable it).

How to run:

    # mocked/backward-compat tests only (no checkpoint, CPU, fast):
    python -m pytest tests/ -v

    # download from HF and run the parity test:
    RELSIM_CHECKPOINT_DIR=thaoshibe/relsim-qwenvl25-lora python -m pytest tests/ -v -k parity

    # or use a local checkpoint you already have:
    RELSIM_CHECKPOINT_DIR=/path/to/relsim_checkpoints python -m pytest tests/ -v -k parity
"""
import os
import types

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

import relsim.relsim_score as rs  # noqa: E402
from relsim.relsim_score import _RelSimClass, preprocess_image  # noqa: E402

EMB_DIM = 384


# --------------------------------------------------------------------------- #
# Lightweight fakes (no model download, CPU only)
# --------------------------------------------------------------------------- #
def _img_key(img):
    """Deterministic per-image seed derived from pixel content."""
    return abs(hash(img.tobytes())) % (2 ** 31)


def _vector_for(seed):
    """Deterministic embedding for a given seed (stable across calls)."""
    g = torch.Generator().manual_seed(int(seed))
    return torch.randn(EMB_DIM, generator=g)


class _FakeParam:
    def __init__(self, device):
        self.device = device


class _FakeBaseModel:
    """
    Mimics QwenWithQueryToken: __call__ returns an unnormalized [B, 384] tensor
    whose rows are a deterministic function of each sample's first input id, so
    output is reproducible and order-checkable. Records call count / batch size
    to prove batching (one forward for the whole batch, not N forwards).
    """

    def __init__(self, device="cpu"):
        self._device = torch.device(device)
        self.call_count = 0
        self.batch_sizes = []
        self.eval_called = False

    def parameters(self):
        yield _FakeParam(self._device)

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, input_ids=None, attention_mask=None,
                 pixel_values=None, image_grid_thw=None, **kwargs):
        self.call_count += 1
        b = input_ids.shape[0]
        self.batch_sizes.append(b)
        rows = [_vector_for(int(input_ids[i, 0].item())) for i in range(b)]
        return torch.stack(rows, dim=0)


class _FakeProcessor:
    """Returns pre-batched, padded tensors; encodes each image into input_ids[:,0]."""

    def __init__(self):
        self.tokenizer = types.SimpleNamespace(pad_token_id=0)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "TEXT"

    def __call__(self, text=None, images=None, videos=None,
                 padding=True, return_tensors="pt"):
        imgs = images if images is not None else []
        b = len(imgs)
        seq_len = 5
        input_ids = torch.ones((b, seq_len), dtype=torch.long)
        for i, img in enumerate(imgs):
            input_ids[i, 0] = _img_key(img)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones((b, seq_len), dtype=torch.long),
            "pixel_values": torch.zeros((b, 3)),
            "image_grid_thw": torch.ones((b, 3), dtype=torch.long),
        }


@pytest.fixture
def model(monkeypatch):
    # Bypass qwen_vl_utils: return the (already-RGB) PIL image from the message.
    monkeypatch.setattr(
        rs, "process_vision_info",
        lambda messages: ([messages[0]["content"][0]["image"]], None),
    )
    return _RelSimClass(_FakeBaseModel(), _FakeProcessor())


def _rgb(color, size=(8, 8)):
    return Image.new("RGB", size, color=color)


# Real sample images shipped in tests/images/ (test_1.jpg, test_2.jpg, ...).
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")
IMAGE_PATHS = sorted(
    os.path.join(IMAGES_DIR, f)
    for f in (os.listdir(IMAGES_DIR) if os.path.isdir(IMAGES_DIR) else [])
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))
)
requires_images = pytest.mark.skipif(
    len(IMAGE_PATHS) == 0, reason="no sample images in tests/images/"
)


@pytest.fixture
def real_images():
    """Load the sample images from tests/images/ as PIL Images."""
    return [Image.open(p) for p in IMAGE_PATHS]


# --------------------------------------------------------------------------- #
# preprocess_image
# --------------------------------------------------------------------------- #
def test_preprocess_single_returns_rgb_image():
    out = preprocess_image(Image.new("L", (4, 4)))
    assert isinstance(out, Image.Image)
    assert out.mode == "RGB"


def test_preprocess_list_returns_list_of_rgb():
    out = preprocess_image([Image.new("L", (4, 4)), Image.new("RGBA", (4, 4))])
    assert isinstance(out, list) and len(out) == 2
    assert all(im.mode == "RGB" for im in out)


def test_preprocess_invalid_type_raises():
    with pytest.raises(TypeError):
        preprocess_image("not-an-image")


# --------------------------------------------------------------------------- #
# embed: shape contract (backward compatible)
# --------------------------------------------------------------------------- #
def test_embed_single_returns_1xD(model):
    out = model.embed(_rgb("red"))
    assert out.shape == (1, EMB_DIM)


def test_embed_list_returns_NxD(model):
    out = model.embed([_rgb("red"), _rgb("green"), _rgb("blue")])
    assert out.shape == (3, EMB_DIM)


def test_embed_single_uses_one_forward_batch1(model):
    model.embed(_rgb("red"))
    assert model.base_model.call_count == 1
    assert model.base_model.batch_sizes == [1]


def test_embed_list_uses_one_forward_not_n_calls(model):
    """Key regression guard: a batch must be ONE forward, not one-per-image."""
    model.embed([_rgb("red"), _rgb("green"), _rgb("blue")])
    assert model.base_model.call_count == 1
    assert model.base_model.batch_sizes == [3]


# --------------------------------------------------------------------------- #
# embed: correctness / consistency
# --------------------------------------------------------------------------- #
def test_embed_output_is_l2_normalized(model):
    out = model.embed([_rgb("red"), _rgb("green")])
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_embed_single_matches_single_in_list(model):
    img = _rgb("red")
    single = model.embed(img)          # [1, D]
    in_list = model.embed([img])       # [1, D]
    assert torch.allclose(single, in_list, atol=1e-6)


def test_embed_batch_order_preserved_and_matches_individual(model):
    a, b, c = _rgb("red"), _rgb("green"), _rgb("blue")
    batched = model.embed([a, b, c])
    ea = model.embed(a)[0]
    eb = model.embed(b)[0]
    ec = model.embed(c)[0]
    assert torch.allclose(batched[0], ea, atol=1e-6)
    assert torch.allclose(batched[1], eb, atol=1e-6)
    assert torch.allclose(batched[2], ec, atol=1e-6)


def test_embed_batch_order_is_strict_with_duplicates(model):
    """A transposition would be caught: row0 and row2 (same image) must match,
    and differ from row1 (different image)."""
    a, b = _rgb("red"), _rgb("green")
    out = model.embed([a, b, a])           # deliberately a, b, a
    assert torch.allclose(out[0], out[2], atol=1e-6)      # both are `a`
    assert not torch.allclose(out[0], out[1], atol=1e-3)  # `a` != `b`
    assert torch.allclose(out[0], model.embed(a)[0], atol=1e-6)
    assert torch.allclose(out[1], model.embed(b)[0], atol=1e-6)


def test_embed_nonpil_element_raises(model):
    with pytest.raises(TypeError):
        model.embed([_rgb("red"), "not-an-image"])


def test_embed_micro_batching_matches_single_forward(model):
    imgs = [_rgb((i, 0, 0)) for i in range(5)]
    full = model.embed(imgs)                          # one forward
    model.base_model.call_count = 0
    model.base_model.batch_sizes = []
    chunked = model.embed(imgs, micro_batch_size=2)   # 3 forwards (2+2+1)
    assert model.base_model.batch_sizes == [2, 2, 1]
    assert torch.allclose(full, chunked, atol=1e-6)


# --------------------------------------------------------------------------- #
# __call__: similarity contract (backward compatible)
# --------------------------------------------------------------------------- #
def test_call_returns_float_similarity(model):
    sim = model(_rgb("red"), _rgb("green"))
    assert isinstance(sim, float)


def test_call_equals_dot_of_embeddings(model):
    a, b = _rgb("red"), _rgb("green")
    sim = model(a, b)
    expected = (model.embed(a)[0] * model.embed(b)[0]).sum().item()
    assert abs(sim - expected) < 1e-5


def test_call_self_similarity_is_one(model):
    a = _rgb("red")
    assert abs(model(a, a) - 1.0) < 1e-5


def test_call_uses_single_batched_forward(model):
    model(_rgb("red"), _rgb("green"))
    assert model.base_model.call_count == 1
    assert model.base_model.batch_sizes == [2]


# --------------------------------------------------------------------------- #
# Real sample images (tests/images/) through the load + preprocess path.
# Still uses the fake model, so no checkpoint/GPU needed — this exercises real
# JPEG decoding, preprocess_image, and batched embed orchestration.
# --------------------------------------------------------------------------- #
@requires_images
def test_preprocess_real_images_returns_rgb():
    out = preprocess_image([Image.open(p) for p in IMAGE_PATHS])
    assert isinstance(out, list) and len(out) == len(IMAGE_PATHS)
    assert all(im.mode == "RGB" for im in out)


@requires_images
def test_embed_real_images_batch(model, real_images):
    imgs = preprocess_image(real_images)
    out = model.embed(imgs)
    assert out.shape == (len(imgs), EMB_DIM)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    # one batched forward over all images
    assert model.base_model.call_count == 1
    assert model.base_model.batch_sizes == [len(imgs)]


@requires_images
def test_embed_real_images_order_matches_individual(model, real_images):
    imgs = preprocess_image(real_images)
    batched = model.embed(imgs)
    for i, im in enumerate(imgs):
        assert torch.allclose(batched[i], model.embed(im)[0], atol=1e-6)


# --------------------------------------------------------------------------- #
# Real-model parity (skipped unless a checkpoint + CUDA are available)
# --------------------------------------------------------------------------- #
# Which checkpoint the real parity test uses:
#   * RELSIM_CHECKPOINT_DIR set  -> that local path or HF id
#   * RELSIM_RUN_REAL=1          -> download the public model from HF
#   * neither                    -> skip (keeps CI light; ~18 GB download otherwise)
_REAL_CKPT = os.environ.get("RELSIM_CHECKPOINT_DIR") or (
    "thaoshibe/relsim-qwenvl25-lora" if os.environ.get("RELSIM_RUN_REAL") else None
)


@requires_images
@pytest.mark.skipif(
    _REAL_CKPT is None,
    reason="set RELSIM_CHECKPOINT_DIR (local path or HF id), or RELSIM_RUN_REAL=1 "
           "to download from HF; runs on CPU or CUDA",
)
def test_real_single_vs_batch_parity():
    from relsim.relsim_score import relsim as load_relsim

    m, preprocess = load_relsim(pretrained=True, checkpoint_dir=_REAL_CKPT)

    imgs = preprocess([Image.open(p) for p in IMAGE_PATHS])
    batched = m.embed(imgs)
    single = torch.cat([m.embed(im) for im in imgs], dim=0)

    assert batched.shape == (len(imgs), EMB_DIM)
    cos = F.cosine_similarity(batched, single, dim=-1)
    assert cos.min().item() > 0.999
