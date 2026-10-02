"""Experiment-specific PrefEval full-state contracts; no model or I/O dependencies."""
from __future__ import annotations

import hashlib
import json
import re

SEED = 20260917
ABSENT = "no active preference"
MAX_RECOVERY_TOKENS = 128
OUTPUT_CONTRACT_V2 = (
    "Return only the complete stored preference value for the requested scope. "
    "The scope label is not part of the value. Preserve every word and any quotation "
    "marks already present in the stored value. Do not add a label, quotation marks, "
    "tags, or explanation. For an inactive preference, return exactly: no active preference"
)
TRAIN_STEMS_V2 = ("What is my current {scope} preference?",
    "Retrieve my active preference for {scope}.",
    "Use the memory to report my {scope} preference verbatim, including every condition.")
HELDOUT_STEMS_V2 = ("Which complete preference statement is currently stored for {scope}?",
    "Read back the entire active {scope} preference exactly as recorded.")


def queries_v2(state, *, training=False):
    stems = TRAIN_STEMS_V2 if training else HELDOUT_STEMS_V2
    return [{"scope": scope, "form": i,
             "query": stem.format(scope=scope_name(scope)) + "\n" + OUTPUT_CONTRACT_V2,
             "target": value if value is not None else ABSENT, "kind": "recovery"}
            for scope, value in sorted(state.items()) for i, stem in enumerate(stems)]


def text_prefix_v2(state):
    entries = [f"Scope: {scope_name(scope)}\n<stored_value>\n{value if value is not None else ABSENT}\n</stored_value>"
               for scope, value in sorted(state.items())]
    return ("Current memory values follow. Scope labels and <stored_value> delimiters "
            "are serialization, not part of the stored values.\n" + "\n".join(entries) + "\n\n")
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


def recovery_summary(state, expected_queries, rows, png_sha):
    """Validate exact registered cells before any whole-image conjunction.

    Callers reconstruct each row's correctness from raw text/tokens. This helper
    never turns missing or repeated successful observations into coverage.
    """
    from collections import Counter,defaultdict
    expected=Counter(digest(q) for q in expected_queries)
    observed=Counter(digest(r['query']) for r in rows)
    if expected!=observed or any(n!=1 for n in observed.values()):
        raise ValueError('Missing, duplicate or altered registered query')
    if not png_sha or any(r.get('png_sha256')!=png_sha for r in rows):
        raise ValueError('Read is not bound to the same reopened PNG')
    recovery=[r for r in rows if r['query']['kind']=='recovery']
    cells={(r['query']['scope'],r['query']['form']) for r in recovery}
    if cells!={(scope,form) for scope in state for form in (0,1)} or len(recovery)!=2*len(state):
        raise ValueError('Incomplete slot/form coverage')
    for row in recovery:
        q=row['query'];value=state[q['scope']]
        if q['target']!=(value if value is not None else ABSENT):
            raise ValueError('Registered state/answer mismatch')
    auxiliary=defaultdict(lambda:[0,0])
    for row in rows:
        if row['query']['kind']!='recovery':
            pair=auxiliary[row['query']['kind']];pair[0]+=bool(row['score']['strict_correct']);pair[1]+=1
    return dict(recovery_complete=all(r['score']['strict_correct'] for r in recovery),
                auxiliary_family_counts=dict(auxiliary),
                all_registered_checks_pass=all(r['score']['strict_correct'] for r in rows))
