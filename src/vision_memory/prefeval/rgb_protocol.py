"""Experiment-specific PrefEval full-state contracts; no model or I/O dependencies."""
from __future__ import annotations

import hashlib
import json
import re

SEED = 20260917
ABSENT = "no active preference"
MAX_RECOVERY_TOKENS = 128
TRAIN_FORMS = (
    "What is my current {scope} preference? Repeat the complete stored statement exactly, or say 'no active preference' if it was cleared. Return only that answer.",
    "Retrieve my active preference for {scope}. Quote the full original statement without quotation marks; if none is active, answer 'no active preference'.",
    "Use the memory to report my {scope} preference verbatim, including every condition. Output only the stored statement or 'no active preference'.",
)
HELDOUT_FORMS = (
    "Which complete preference statement is currently stored for {scope}? Reproduce it verbatim, with no explanation. If it has been removed, return 'no active preference'.",
    "Read back the entire active {scope} preference exactly as recorded. Give no other text. For a cleared preference, respond 'no active preference'.",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def normalize(text):
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def scope_name(topic):
    return topic.replace("_", " ")


def state_id(state):
    return "state-" + digest(state)[:20]


def queries(state, *, training=False):
    forms = TRAIN_FORMS if training else HELDOUT_FORMS
    return [{"scope": scope, "form": i, "query": template.format(scope=scope_name(scope)),
             "target": value if value is not None else ABSENT, "kind": "recovery"}
            for scope, value in sorted(state.items()) for i, template in enumerate(forms)]


def transition(state, scope, value, operation):
    if operation not in ("set", "overwrite", "clear", "retain"):
        raise ValueError(operation)
    result = dict(state)
    if operation == "set":
        if scope in result and result[scope] is not None:
            raise ValueError("SET cannot silently overwrite an active unit")
        result[scope] = value
    elif operation == "overwrite":
        if result.get(scope) is None or not value:
            raise ValueError("OVERWRITE requires an active predecessor and replacement")
        result[scope] = value
    elif operation == "clear":
        if result.get(scope) is None:
            raise ValueError("CLEAR requires an active preference")
        result[scope] = None
    elif scope is not None or value is not None:
        raise ValueError("RETAIN has no privileged state payload")
    if operation in ("set", "overwrite") and not value:
        raise ValueError("Missing preference disclosure")
    return result


def event_wordings(scope, value, operation):
    name = scope_name(scope) if scope else ""
    if operation == "set":
        return [f"Remember my {name} preference: {value}",
                f"Store the following preference for {name}: {value}"]
    if operation == "overwrite":
        return [f"Replace my previous {name} preference with this one: {value}",
                f"Update only my {name} preference to: {value}"]
    if operation == "clear":
        return [f"Forget my {name} preference. Keep all other preferences unchanged.",
                f"Remove only the stored preference for {name}; leave it unspecified."]
    return ["No preferences have changed. Keep all stored preferences as they are.",
            "Retain my current preferences without adding or removing any."]


def writer_input(previous_png, event):
    """This is the entire semantic model boundary; IDs/labels never enter it."""
    if not isinstance(event, str) or not event.strip():
        raise ValueError("Expected event text")
    return {"image": previous_png, "event": event}


def loss_groups(state, changed_scope):
    changed = [changed_scope] if changed_scope in state else []
    unchanged = [key for key in sorted(state) if key not in changed]
    return changed, unchanged


def balanced_state_loss(losses, changed_scope):
    """Each present category has equal weight, regardless of its unit count."""
    changed, unchanged = loss_groups(losses, changed_scope)
    groups = [sum(losses[k] for k in keys) / len(keys)
              for keys in (changed, unchanged) if keys]
    if not groups:
        raise ValueError("A teacher target requires at least one addressed unit")
    return sum(groups) / len(groups)


def official_mcq_valid(original_preference, current_preference):
    return original_preference == current_preference and current_preference is not None
