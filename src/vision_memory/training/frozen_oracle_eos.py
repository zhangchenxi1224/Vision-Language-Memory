"""Open-answer contracts for fresh Frozen DreamLite input optimization."""
from __future__ import annotations
import re

INSTRUCTIONS = "Use the memory image to answer.\nAnswer with a short phrase only."


def question_prompts(query: str) -> dict[str, str]:
    original = query.removesuffix(" Choose exactly one option.")
    match = re.fullmatch(r"(?P<prefix>R3 Train Standard Templates \d+: )(?P<time>At this later check|After applying this update first|Before any later update), what is the current (?P<category>.+?) preference for the (?P<entity>.+?)\?", original)
    if not match:
        raise ValueError(f"Unrecognized locked question: {query!r}")
    prefix, category, entity = (match[k] for k in ("prefix", "category", "entity"))
    time = match['time']
    later = time[0].lower() + time[1:]
    questions = {
        "original_open": original,
        "paraphrase_1": prefix + f"{time}, which {category} does the {entity} currently prefer?",
        "paraphrase_2": prefix + f"What {category} does the {entity} prefer now, {later}?",
        "paraphrase_3": prefix + f"For the {entity}, what is the {category} preference currently in effect {later}?",
        "paraphrase_4": prefix + f"Based on the stored memory, name the current preferred {category} for the {entity} {later}.",
    }
    return {key: question + "\n" + INSTRUCTIONS for key, question in questions.items()}


def endpoint_gate(rows):
    matched = [r for r in rows if r['condition'] == 'matched']
    if len(matched) != 5 or {r['prompt_id'] for r in matched} != {'original_open', 'paraphrase_1', 'paraphrase_2', 'paraphrase_3', 'paraphrase_4'}:
        raise ValueError('Endpoint must contain all five fixed-instruction questions')
    original = next(r for r in matched if r['prompt_id'] == 'original_open')
    return {'qa_pass': bool(original['strict_correct']), 'gate': 'original_open_raw_greedy32_exact_match_step256',
            'normal_correct': sum(bool(r['strict_correct']) for r in matched),
            'paraphrase_all_correct': all(r['strict_correct'] for r in matched),
            'answer_prefix_correct': bool(original['answer_prefix_token_exact']),
            'overgeneration': bool(original['overgeneration']),
            'blank_correct': sum(bool(r['strict_correct']) for r in rows if r['condition'] == 'blank'),
            'donor_correct': sum(bool(r['strict_correct']) for r in rows if r['condition'] == 'donor'),
            'legacy_reachability_gate': None}
