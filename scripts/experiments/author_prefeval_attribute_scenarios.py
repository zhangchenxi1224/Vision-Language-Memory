"""Create the fixed Plan10 attribute-generalization application panel.

The source is the already sanitized preference-only audit.  Each value gets two
new decision situations and a counterfactual version of each situation.  The
counterfactual keeps proposal identities fixed but moves the decisive attribute
to a different proposal, so memorising an answer position or provider name is
insufficient.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "reports/prefeval-rgb-20260917/semantic-transfer-v1"
OUT = ROOT / "reports/prefeval-rgb-20260917/attribute-generalization-v1"
INSTRUCTION = (
    "Select the proposal that satisfies the stored preference relevant to this "
    "situation. Return only the complete selected proposal text, without its "
    "label or any explanation."
)
NAMES = ("North proposal", "East proposal", "South proposal", "West proposal")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _proposal(name, context, attribute):
    return f"{name} {context}; its binding specification says {attribute}."


def _case(value, audit, family, situation_index, variant, permutation):
    attributes = [x["attributes"] for x in audit["options"]]
    contexts = (
        (
            "A coordinator must commit to one concrete plan for an upcoming "
            "personal decision. The four specifications below are complete and "
            "authoritative. Which named proposal should be committed to using "
            "the relevant stored preference?",
            "would be used for the upcoming commitment",
        ),
        (
            "A reviewer must approve exactly one provider policy for repeated "
            "future use. Only the declared properties below may be considered. "
            "Which complete policy should be approved using the relevant stored "
            "preference?",
            "would govern every future use under the approved policy",
        ),
    )
    situation, context = contexts[situation_index]
    proposals = [
        _proposal(name, context, attributes[permutation[i]])
        for i, name in enumerate(NAMES)
    ]
    source_correct = audit["correct_option"]
    correct = permutation.index(source_correct)
    case_id = f"{family}:{situation_index}:{variant}"
    return dict(
        id=case_id,
        value_id=value["id"],
        scope=value["scope"],
        scenario=situation_index * 2 + variant,
        situation=situation,
        proposals=proposals,
        target=proposals[correct],
        correct_option=correct,
        instruction=INSTRUCTION,
        base_rotation=int(digest(case_id)[:8], 16) % 4,
        situation_family=situation_index,
        counterfactual_variant=variant,
        attribute_permutation=list(permutation),
    )


def build():
    source = json.loads((OLD / "authoring-source.json").read_text(encoding="utf-8"))
    audits = json.loads((OLD / "clause-option-audit.json").read_text(encoding="utf-8"))
    values = {x["id"]: x for x in source["values"]}
    cases = {}
    audit_out = {}
    # Variant one moves the uniquely compatible attribute from identity 0 to 2;
    # the second situation uses identities 1 and 3.  The same permutations are
    # shared by overwrite contrasts, preserving identical questions/options.
    permutations = (((0, 1, 2, 3), (3, 0, 1, 2)), ((2, 3, 0, 1), (1, 2, 3, 0)))
    for vid, value in values.items():
        source_audit = audits[vid]
        family = digest(["attribute-generalization", vid])[:12]
        built = []
        for situation_index in range(2):
            for variant in range(2):
                built.append(
                    _case(
                        value,
                        source_audit,
                        family,
                        situation_index,
                        variant,
                        permutations[situation_index][variant],
                    )
                )
        assert len(built) == 4
        for left, right in ((built[0], built[1]), (built[2], built[3])):
            assert left["situation"] == right["situation"]
            assert left["correct_option"] != right["correct_option"]
            assert left["proposals"] != right["proposals"]
        cases[vid] = built
        audit_out[vid] = dict(
            source=value["value"],
            scope=value["scope"],
            clauses=source_audit["clauses"],
            source_option_audit=source_audit["options"],
            cases=[
                dict(
                    id=x["id"],
                    correct_option=x["correct_option"],
                    attribute_permutation=x["attribute_permutation"],
                )
                for x in built
            ],
            authoring_inputs=(
                "sanitized preference clauses and fictional option attributes; "
                "no benchmark questions, answers, explanations, model outputs, "
                "or failure records"
            ),
        )
    for contrast in source["overwrite_contrasts"]:
        before, after = cases[contrast["before"]], cases[contrast["after"]]
        for a, b in zip(before, after):
            assert a["situation"] == b["situation"]
            assert a["proposals"] == b["proposals"]
            assert a["correct_option"] != b["correct_option"]
    return source, cases, audit_out


def main():
    source, cases, audit = build()
    OUT.mkdir(parents=True, exist_ok=True)
    payloads = {
        "authoring-source.json": source,
        "attribute-scenarios.json": cases,
        "attribute-scenario-audit.json": audit,
    }
    for name, payload in payloads.items():
        path = OUT / name
        if path.exists():
            assert json.loads(path.read_text(encoding="utf-8")) == payload
        else:
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
                newline="\n",
            )
    print(json.dumps(dict(values=len(cases), cases=sum(map(len, cases.values())), overwrite_contrasts=len(source["overwrite_contrasts"]))))


if __name__ == "__main__":
    main()
