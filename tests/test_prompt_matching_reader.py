"""CPU contracts for exact-continuation, frozen-Reader prompt matching."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vision_memory.reader.open_answer import generate_short_answer
from vision_memory.reader.prompt_matching import (
    qwen3vl_continuation_logits,
    soft_target_cross_entropy,
    soft_target_kl_divergence,
    validate_generated_target_ids,
)


class FakeProcessor:
    def __init__(self, *, locked=False, detach=False, bad_grid=False):
        self.locked, self.detach, self.bad_grid = locked, detach, bad_grid
        self.tokenizer = SimpleNamespace(eos_token_id=100)

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        self.messages = deepcopy(messages)
        return "USER IMAGE " + messages[0]["content"][1]["text"] + " ASSISTANT"

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        pixels = kwargs["images"][0] * 1.0
        if self.locked:
            assert tuple(pixels.shape) == (3, 256, 256)
            pixels = pixels.reshape(-1).repeat(2).reshape(256, 1536).float()
        if self.detach:
            pixels = pixels.detach()
        self.batch = {
            # This includes image expansion, so prefix length cannot be guessed from text.
            "input_ids": torch.tensor([[31, 32, 33, 34, 35]]),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
            "mm_token_type_ids": torch.tensor([[0, 1, 1, 0, 0]]),
            "pixel_values": pixels,
            "image_grid_thw": torch.tensor([[1, 8 if self.bad_grid else 16, 16]]),
        }
        return self.batch

    def batch_decode(self, ids, **kwargs):
        # Only generation uses decoding. The scorer must not call this method.
        self.decode_calls = getattr(self, "decode_calls", 0) + 1
        return ["answer" if kwargs["skip_special_tokens"] else "answer<|im_end|>"]


class FakeReader(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head = torch.nn.Linear(3, 128, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.arange(384).reshape(128, 3).float().sin() / 5)
        self.generation_config = SimpleNamespace(eos_token_id=[99, 100])
        self.eval().requires_grad_(False)

    def model(self, **kwargs):
        self.observed = kwargs
        ids = kwargs["input_ids"]
        positions = torch.arange(ids.shape[1], device=ids.device).float().expand_as(ids)
        # Causal states depend only on the prefix seen so far, plus visual input.
        hidden = torch.stack((positions, ids.float().cumsum(1) / 100,
                              kwargs["pixel_values"].mean().expand_as(positions)), dim=-1)
        self.full_hidden = hidden
        return SimpleNamespace(last_hidden_state=hidden)

    def generate(self, **kwargs):
        self.generate_observed = kwargs
        return SimpleNamespace(sequences=torch.cat((kwargs["input_ids"], torch.tensor([[11, 12, 100]])), dim=1))


def score(*, reader=None, processor=None, image=None, **kwargs):
    return qwen3vl_continuation_logits(
        model=reader if reader is not None else FakeReader(),
        processor=processor if processor is not None else FakeProcessor(),
        image=image if image is not None else torch.full((3, 4, 4), .4, requires_grad=True),
        query=kwargs.pop("query", "What is my preference?"),
        target_ids=kwargs.pop("target_ids", (11, 12, 100)), device="cpu",
        reader_resize_contract=kwargs.pop("reader_resize_contract", None), **kwargs,
    )


def receipt():
    return {
        "generated_token_ids": [11, 12, 100], "generated_token_count": 3,
        "eos_token_ids": [99, 100], "eos_reached": True,
        "truncated": False, "finish_reason": "eos",
    }


def test_exact_ids_causal_offset_and_prefix_metadata():
    reader, processor = FakeReader(), FakeProcessor()
    output = score(reader=reader, processor=processor)
    assert reader.observed["input_ids"].tolist() == [[31, 32, 33, 34, 35, 11, 12, 100]]
    assert reader.observed["mm_token_type_ids"].tolist() == [[0, 1, 1, 0, 0, 0, 0, 0]]
    assert reader.observed["attention_mask"].tolist() == [[1] * 8]
    assert reader.observed["use_cache"] is False
    assert processor.batch["input_ids"].tolist() == [[31, 32, 33, 34, 35]]
    assert processor.batch["mm_token_type_ids"].shape == (1, 5)  # no input mutation
    torch.testing.assert_close(output.target_logits, reader.lm_head(reader.full_hidden[:, 4:7]))
    torch.testing.assert_close(output.loss, F.cross_entropy(output.target_logits.reshape(-1, 128),
                                                         torch.tensor([11, 12, 100])))
    assert output.target_ids.tolist() == [[11, 12, 100]]
    assert processor.messages == [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": "What is my preference?"},
    ]}]
    assert processor.kwargs["text"] == ["USER IMAGE What is my preference? ASSISTANT"]
    assert processor.kwargs["do_rescale"] is False
    assert getattr(processor, "decode_calls", 0) == 0


def test_first_logit_does_not_observe_first_continuation_token():
    output_a = score(target_ids=[11, 12, 100])
    output_b = score(target_ids=[17, 12, 100])
    torch.testing.assert_close(output_a.target_logits[:, 0], output_b.target_logits[:, 0])
    assert not torch.equal(output_a.target_logits[:, 1], output_b.target_logits[:, 1])


def test_generation_and_scoring_use_identical_prompt_and_supplied_token_ids():
    reader, processor = FakeReader(), FakeProcessor()
    query = "Earlier exchange: user: I like tea. assistant: Understood.\nCurrent question: What drink?"
    generated = generate_short_answer(model=reader, processor=processor, image=torch.zeros(3, 4, 4),
                                      query=query, device="cpu", reader_resize_contract=None)
    ids = validate_generated_target_ids(generated, assistant_end_token_id=100, pad_token_id=0)
    calls_before = processor.decode_calls
    with torch.no_grad():
        output = score(reader=reader, processor=processor, image=torch.zeros(3, 4, 4), query=query,
                       target_ids=ids, require_image_grad=False)
    assert processor.kwargs["text"] == [generated["chat_prompt"]]
    assert reader.observed["input_ids"][0, :5].tolist() == generated["input_token_ids"]
    assert output.target_ids[0].tolist() == generated["generated_token_ids"]
    assert processor.decode_calls == calls_before
    assert not output.target_logits.requires_grad


def test_student_image_gradient_and_no_reader_parameter_gradient():
    reader = FakeReader()
    image = torch.full((3, 4, 4), .4, requires_grad=True)
    output = score(reader=reader, image=image)
    output.loss.backward()
    assert image.grad is not None and torch.isfinite(image.grad).all() and image.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in reader.parameters())
    assert not reader.training


def test_locked_resize_uses_existing_contract_and_rejects_grid_drift():
    processor = FakeProcessor(locked=True)
    image = torch.full((3, 1024, 1024), .4)
    from vision_memory.reader.qwen3vl import R3_QWEN_READER_RESIZE_CONTRACT
    with torch.no_grad():
        output = score(processor=processor, image=image, require_image_grad=False,
                       reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT)
        assert processor.kwargs["do_resize"] is False
        assert output.pixel_values.shape == (256, 1536)
        with pytest.raises(RuntimeError, match="grid drifted"):
            score(processor=FakeProcessor(locked=True, bad_grid=True), image=image, require_image_grad=False,
                  reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT)


def test_detached_image_or_grad_disabled_student_fails():
    with pytest.raises(RuntimeError, match="detached"):
        score(processor=FakeProcessor(detach=True))
    with torch.no_grad(), pytest.raises(RuntimeError, match="detached"):
        score()


def test_reader_frozen_eval_is_required_without_mutation():
    reader = FakeReader().train()
    with pytest.raises(ValueError, match="eval-mode"):
        score(reader=reader)
    assert reader.training
    reader.eval().requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        score(reader=reader)
    assert reader.lm_head.weight.requires_grad
    reader.requires_grad_(False)
    reader.lm_head.train()
    with pytest.raises(ValueError, match="submodules"):
        score(reader=reader)


@pytest.mark.parametrize("ids", [[], [True], [-1], [1.5], "answer", torch.tensor([1.0]),
                                 torch.tensor([[1], [2]]), torch.tensor([], dtype=torch.long)])
def test_invalid_continuation_ids_fail(ids):
    with pytest.raises((TypeError, ValueError)):
        score(target_ids=ids)


def test_invalid_vocab_or_query_fail():
    with pytest.raises(ValueError, match="vocabulary"):
        score(target_ids=[129])
    with pytest.raises(ValueError, match="nonempty"):
        score(query=" ")


def test_soft_ce_uses_full_vocabulary_fp32_and_detaches_teacher():
    student = torch.tensor([[[1., -1., 2., .3], [.5, 1., -.5, 0.]]], dtype=torch.float16, requires_grad=True)
    teacher = torch.tensor([[[2., 0., -.2, .4], [.5, -.2, 1., .3]]], dtype=torch.float16, requires_grad=True)
    loss = soft_target_cross_entropy(student, teacher, temperature=2.)
    expected = -(F.softmax(teacher.detach().float() / 2., -1)
                 * F.log_softmax(student.float() / 2., -1)).sum(-1).mean() * 4.
    torch.testing.assert_close(loss, expected)
    assert loss.dtype == torch.float32
    loss.backward()
    assert teacher.grad is None
    assert student.grad is not None and torch.all(student.grad != 0)


def test_t1_ce_minus_entropy_equals_kl_and_same_student_gradient():
    student = torch.tensor([[.2, -.7, 1.], [1., .3, -.2]], requires_grad=True)
    teacher = torch.tensor([[.7, 1., -.2], [-1., .2, .8]], requires_grad=True)
    ce = soft_target_cross_entropy(student, teacher)
    kl = soft_target_kl_divergence(student, teacher)
    logp = F.log_softmax(teacher.detach(), -1)
    entropy = -(logp.exp() * logp).sum(-1).mean()
    torch.testing.assert_close(ce - entropy, kl)
    ce_grad, = torch.autograd.grad(ce, student, retain_graph=True)
    kl_grad, = torch.autograd.grad(kl, student)
    torch.testing.assert_close(ce_grad, kl_grad)
    assert teacher.grad is None
    torch.testing.assert_close(soft_target_kl_divergence(teacher.detach(), teacher), torch.tensor(0.))


def test_token_mask_excludes_only_selected_positions_and_normalizes_selected_count():
    student = torch.tensor([[[0., 2., -1.], [1., 0., 2.], [-1., 1., 0.]]], requires_grad=True)
    teacher = torch.tensor([[[2., 0., 1.], [0., 1., 2.], [2., 1., 0.]]], requires_grad=True)
    mask = torch.tensor([[True, False, True]])
    loss = soft_target_cross_entropy(student, teacher, mask=mask)
    expected = soft_target_cross_entropy(student[:, [0, 2]], teacher[:, [0, 2]])
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.equal(student.grad[:, 1], torch.zeros_like(student.grad[:, 1]))
    assert student.grad[:, [0, 2]].abs().sum() > 0
    assert teacher.grad is None


@pytest.mark.parametrize("temperature", [0., -1., float("nan"), float("inf"), True, "1", 1e200, 1e20, 1e-50])
def test_invalid_temperature_fails(temperature):
    with pytest.raises(ValueError, match="temperature"):
        soft_target_cross_entropy(torch.zeros(1, 2), torch.zeros(1, 2), temperature=temperature)


@pytest.mark.parametrize("mask", [torch.zeros(1, 2), torch.ones(2), torch.tensor([[1., .5]]),
                                 torch.tensor([[1., float("nan")]])])
def test_invalid_masks_fail(mask):
    with pytest.raises(ValueError, match="mask"):
        soft_target_cross_entropy(torch.zeros(1, 2, 3), torch.zeros(1, 2, 3), mask=mask)


@pytest.mark.parametrize("student,teacher", [
    (torch.zeros(1, 2), torch.zeros(1, 3)),
    (torch.zeros(2), torch.zeros(2)),
    (torch.zeros(0, 2), torch.zeros(0, 2)),
    (torch.zeros(1, 2, dtype=torch.long), torch.zeros(1, 2)),
    (torch.tensor([[float("nan"), 0.]]), torch.zeros(1, 2)),
    (torch.zeros(1, 2), torch.tensor([[0., float("inf")]])),
])
def test_invalid_soft_targets_fail(student, teacher):
    with pytest.raises(ValueError):
        soft_target_cross_entropy(student, teacher)


def test_eos_validation_returns_exact_unmodified_ids():
    original = receipt()
    snapshot = deepcopy(original)
    assert validate_generated_target_ids(original, assistant_end_token_id=100, pad_token_id=0) == (11, 12, 100)
    assert original == snapshot
    assert validate_generated_target_ids(original, assistant_end_token_id=100, pad_token_id=100) == (11, 12, 100)


@pytest.mark.parametrize("changes", [
    {"generated_token_ids": [11, 12], "generated_token_count": 2},
    {"generated_token_ids": [11, 12, 99]},  # another configured stop is not assistant EOS
    {"generated_token_ids": [11, 100, 0]},
    {"generated_token_ids": [11, 99, 100]},
    {"generated_token_ids": [11, 0, 100]},
    {"generated_token_ids": [11, True, 100]},
    {"generated_token_ids": []},
    {"generated_token_count": 2},
    {"eos_token_ids": [99]},
    {"eos_reached": False},
    {"truncated": True},
    {"finish_reason": "token_limit"},
])
def test_eos_validation_rejects_truncation_padding_or_inconsistent_receipts(changes):
    generation = receipt() | changes
    with pytest.raises(ValueError):
        validate_generated_target_ids(generation, assistant_end_token_id=100, pad_token_id=0)
