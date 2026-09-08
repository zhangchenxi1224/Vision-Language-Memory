"""CPU-only P0/P4 audit of actual Qwen chat terminator and continuation tokens."""
from pathlib import Path
import argparse
import json
import sys
import hashlib
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reader", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    generation = json.loads((args.reader / "generation_config.json").read_text())
    tokenizer_config = json.loads((args.reader / "tokenizer_config.json").read_text())
    template_text = tokenizer_config["chat_template"]
    template_file = json.loads((args.reader / "chat_template.json").read_text())
    if isinstance(template_file, dict):
        template_file = template_file["chat_template"]
    if template_file != template_text:
        raise ValueError("Tokenizer and processor chat templates differ; use GPU full-processor audit.")
    template = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(template_text)
    tokenizer = Tokenizer.from_file(str(args.reader / "tokenizer.json"))
    user = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Return a short answer."}]}
    def render(messages, generation_prompt):
        return template.render(messages=messages, add_generation_prompt=generation_prompt,
                               tools=None, add_vision_id=False)
    sentinel = "R11EOSBoundaryProbe"
    prefix = render([user], True)
    complete = render([user, {"role": "assistant", "content": sentinel}], False)
    if not complete.startswith(prefix + sentinel):
        raise ValueError("Actual assistant completion does not extend its generation prompt.")
    suffix = complete[len(prefix + sentinel):]
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False).ids
    if not suffix_ids or suffix_ids[0] not in generation["eos_token_id"]:
        raise ValueError("Assistant ending does not match generation stopping contract.")
    config = json.loads((ROOT / "configs/experiments/r11_open_eos_paired.json").read_text())
    examples = []
    for example in config["p4_tokenization_examples"]:
        prompt_user = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text":
            f"State the current {example['field']} preference. Return only a short phrase."}]}
        prompt = render([prompt_user], True)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
        joint_ids = tokenizer.encode(prompt + example["answer"], add_special_tokens=False).ids
        if joint_ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError("Joint BPE tokenization changes prompt prefix.")
        answer_ids = joint_ids[len(prompt_ids):]
        examples.append({**example, "answer_token_ids": answer_ids, "answer_token_count": len(answer_ids)})
    result = {"termination": {"assistant_end_token_id": suffix_ids[0], "assistant_end_token_text": tokenizer.id_to_token(suffix_ids[0]),
              "generation_eos_token_ids": generation["eos_token_id"], "chat_suffix": suffix,
              "chat_suffix_token_ids": suffix_ids, "chat_template_sha256": hashlib.sha256(template_text.encode()).hexdigest()},
              "examples": examples, "audit_runtime": "CPU Rust tokenizer + identical processor/tokenizer Jinja template",
              "full_processor_contract_rechecked_by_gpu_runner": True,
              "model_forward_calls": 0, "optimizer_steps": 0, "scientific_memory_success": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
