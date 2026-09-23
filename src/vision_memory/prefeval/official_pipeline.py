"""PrefEval's released SFT data recipe and full-dialogue benchmark inputs.

This is separate from the historical oracle-sparse/state-recovery adapter.
It prepares data, not trained weights. Source: PrefEval commit 50795054.
"""
from __future__ import annotations

from typing import Any

UPSTREAM_REVISION = "50795054b5ff5f418d2b768a331d71e480f93331"
TOPIC_ORDER = (
    "travel_transportation", "shop_motors", "lifestyle_beauty", "travel_restaurant",
    "shop_fashion", "entertain_shows", "pet_ownership", "lifestyle_fit",
    "entertain_games", "shop_home", "lifestyle_health", "travel_activities",
    "education_learning_styles", "entertain_music_book", "professional_work_location_style",
    "education_resources", "lifestyle_dietary", "shop_technology", "travel_hotel", "entertain_sports",
)
# sklearn train_test_split(TOPIC_ORDER, test_size=.2, random_state=42).
# Freeze the resulting order without adding sklearn to the runtime dependencies.
SPLIT_PERMUTATION = (0, 17, 15, 1, 8, 5, 11, 3, 18, 16, 13, 2, 9, 19, 4, 12, 7, 10, 14, 6)
EVAL_TOPICS = tuple(TOPIC_ORDER[i] for i in SPLIT_PERMUTATION[:4])
TRAIN_TOPICS = tuple(TOPIC_ORDER[i] for i in SPLIT_PERMUTATION[4:])
SFT_INTER_TURNS = (0, 5, 10)


def message(role: str, content: str) -> dict[str, str]:
    if role not in ("user", "assistant") or not isinstance(content, str) or not content.strip():
        raise ValueError("Expected a nonempty user/assistant message")
    return {"role": role, "content": content}


def validate_exchanges(messages: list[dict[str, str]]) -> None:
    if len(messages) % 2:
        raise ValueError("Expected complete user/assistant exchanges")
    for i, item in enumerate(messages):
        message(item["role"], item["content"])
        if item["role"] != ("user" if i % 2 == 0 else "assistant"):
            raise ValueError("Messages must alternate user/assistant")


def context_messages(conversations: list[dict], *, for_sft: bool) -> list[dict[str, str]]:
    # train_sft.py uses the last four conversations; common_utils.py uses all.
    selected = conversations[-4:] if for_sft else conversations
    messages = [message(m["role"], m["content"]) for c in selected for m in c["conversation"]]
    validate_exchanges(messages)
    return messages


def context_prefix(messages: list[dict[str, str]], inter_turns: int) -> list[dict[str, str]]:
    if inter_turns < 0 or inter_turns * 2 > len(messages):
        raise ValueError(f"Cannot supply {inter_turns} complete contextual turns")
    selected = [dict(m) for m in messages[:inter_turns * 2]]
    validate_exchanges(selected)
    return selected


def disclosure_messages(row: dict, form: str, *, acknowledgment: str | None = None) -> list[dict[str, str]]:
    if form == "explicit":
        messages = [message("user", row["preference"])]
        if acknowledgment is not None:
            messages.append(message("assistant", acknowledgment))
        return messages
    conversation = row["conversation"]
    if form == "implicit_choice":
        return [message(role, conversation[key]) for role, key in (
            ("user", "query"), ("assistant", "assistant_options"),
            ("user", "user_selection"), ("assistant", "assistant_acknowledgment"),
        )]
    if form == "implicit_persona":
        # Preserve upstream JSON insertion order, as implicit_utils.py does.
        return [message(role, turn[role]) for turn in conversation.values() for role in ("user", "assistant")]
    raise ValueError(f"Unknown preference form: {form}")


def sft_record(row: dict, *, topic: str, row_index: int, contexts: list[dict], inter_turns: int) -> dict:
    if topic not in TOPIC_ORDER or inter_turns not in SFT_INTER_TURNS:
        raise ValueError("Unknown topic or SFT context condition")
    history = disclosure_messages(row, "explicit", acknowledgment=row["response_to_pref"])
    history += context_prefix(contexts, inter_turns)
    validate_exchanges(history)
    target = message("assistant", row["response_to_q"])
    return {
        "base_pair_id": f"{topic}:{row_index:04d}",
        "topic": topic,
        "split": "train" if topic in TRAIN_TOPICS else "eval_topic",
        "inter_turns": inter_turns,
        "system_prompt": "You are an AI assistant.",
        "history": history,
        "query": message("user", row["question"]),
        "target": target,
        "source": {
            "path": f"SFT/single_pref_remind/{topic}/mistral8x7b_{topic}_2turn.json",
            "row_index": row_index, "revision": UPSTREAM_REVISION,
            "teacher_identity": "released_mistral8x7b_filename; paper_describes_Mistral_7B",
        },
    }


def benchmark_generation_messages(
    record: dict, contexts: list[dict], *, inter_turns: int,
    explicit_acknowledgment: str | None = None, max_words: int = 300,
) -> list[dict[str, str]]:
    """Official zero-shot generation conversation, without evaluator metadata.

    Generate explicit_acknowledgment with the model under test before calling
    this function. Benchmark evaluation must not borrow the SFT teacher's reply.
    System prompting is supplied separately by the model-specific runner.
    """
    payload = record["input"]
    history = [dict(m) for m in payload["disclosure"]]
    if payload["needs_model_acknowledgment"]:
        if explicit_acknowledgment is None:
            raise ValueError("Explicit benchmark requires the tested model's acknowledgment")
        history.append(message("assistant", explicit_acknowledgment))
    validate_exchanges(history)
    history += context_prefix(contexts, inter_turns)
    history.append(message("user", payload["query"]["content"] + f" (Please respond within {max_words} words.)"))
    return history


def render_released_mistral(record: dict) -> str:
    """Exact text layout used by the released SFT collator before tokenization."""
    history = record["history"]
    inter = "".join(
        f"[INST] {m['content']} [/INST]" if m["role"] == "user" else f"{m['content']}</s>"
        for m in history[2:]
    )
    return (f"<s>[INST]\n{record['system_prompt']}\n{history[0]['content']}\n[/INST]\n"
            f"{history[1]['content']}</s>\n{inter}\n[INST]\n{record['query']['content']}\n"
            f"[/INST]\n{record['target']['content']}\n</s>")


def writer_training_view(record: dict) -> dict:
    """Query-independent inputs to a shared RGB Writer, plus Reader supervision.

    A consumer must carry only the generated RGB between updates. No latent is
    fitted per record here. This view is an adaptation, not official text SFT.
    """
    history = record["history"]
    validate_exchanges(history)
    return {
        "updates": [[dict(history[i]), dict(history[i + 1])] for i in range(0, len(history), 2)],
        "reader_query": dict(record["query"]),
        "supervision": dict(record["target"]),
    }


def tokenize_answer_only(tokenizer: Any, record: dict, *, max_length: int) -> dict[str, list[int]]:
    """Mask the history and query; supervise only final answer and its end token.

    Use the actual model chat template, not substring offsets in raw text.
    Reject overlength records rather than silently truncating the answer.
    """
    prompt = [{"role": "system", "content": record["system_prompt"]}] + record["history"] + [record["query"]]
    prefix = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(prompt + [record["target"]], tokenize=True, add_generation_prompt=False)
    if full[:len(prefix)] != prefix or len(full) <= len(prefix):
        raise ValueError("Chat template does not provide an exact answer prefix; use a model-specific collator")
    if len(full) > max_length:
        raise ValueError(f"SFT sequence has {len(full)} tokens, exceeding {max_length}; no truncation applied")
    return {"input_ids": full, "attention_mask": [1] * len(full),
            "labels": [-100] * len(prefix) + full[len(prefix):]}
