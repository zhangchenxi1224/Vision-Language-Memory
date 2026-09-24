# Qwen3.8-max judge — 2026-09-24

User authorized replacing the pending generation judge with `qwen3.8-max`.
This authorizes scoring saved answers; new training/diagnostic iterations and
the Goal remain paused. The remote write-ackmix A/B pipelines are now complete.
There were zero remote judge files before this change.

The configured local `DASHSCOPE_API_KEY` successfully accessed the Beijing
Model Studio endpoint. Its model inventory includes `qwen3.8-max`. The first
real saved answer has now received all four judgments, with valid Yes/No XML,
no truncation, and no malformed answers. A cache replay left the judgment file
unchanged and made no additional API calls. This is a connectivity/protocol
check using a real condition, not a performance estimate.

## Frozen judging protocol

- Provider: DashScope OpenAI-compatible Chat Completions.
- Model: `qwen3.8-max` (the user-requested alias; API response identity retained).
- Temperature0, max output100 tokens, thinking disabled, no web search/tools.
- Pinned PrefEval revision `50795054b5ff5f418d2b768a331d71e480f93331`.
- Unmodified official four prompts, XML parsers, and error aggregation.
- Each answer receives acknowledgement, violation, hallucination and helpfulness
  checks; hallucination uses the preference extracted by acknowledgement.
- Save every raw judgment, prompt hash, input hash, response ID and token usage.
- Keep malformed outputs visible; do not silently repair scoring or drop rows.
- Label results **PrefEval protocol with Qwen3.8-max substitute judge**, not an
  exact reproduction of the original Claude3 Sonnet/Bedrock judge.

Use all fixed T1/T2/T3/O1/O2 answers: teacher, original Writer, exact-input
diagnostic, corrected Writer, and blank/full-text/mismatched controls.
The completed corpus contains2,412 condition files and12,060 natural answers:
1,360 original conditions +128 exact-input diagnostic +924 corrected students.
Four checks per answer imply48,240 judgments, including the first cached four.
MCQ scores are unaffected; this is a separate free-answer evaluation.

## Execution and continuation

Run locally against exported answer JSON on CPU; no GPU inference, training or
key transfer is required. Credentials are inherited from the existing environment
and are not written to Git, receipts, command arguments or shared storage.

Run directory: `runs/prefeval-qwen38-judge-20260924` (Git-ignored).
Judge data: `judges/dashscope-qwen3.8-max/` under that directory.
Launch receipt and process identities: `QWEN38_JUDGE_STATUS.json` in this report
directory. Original and corrected corpora use separate, disjoint path globs,
four workers each. Do not duplicate either recorded live process. API/network exceptions
cancel queued work; completed individual checks are retained for explicit resume.

```powershell
D:\st_python\python.exe scripts/eval/judge_prefeval_official_rgb.py `
  --run runs/prefeval-qwen38-judge-20260924 `
  --upstream .cache/prefeval-upstream `
  --provider dashscope --model qwen3.8-max --workers 4 `
  --glob '*/*/write/*.json'
```

The corrected corpus uses the same command with
`--glob 'students/*/write-ackmix/*.json'`; the two selections do not overlap.

After completion, report paired scores by stage, arm, split, image condition and
question form. Do not present the mixed overall corpus average as a method score.
Archive all raw judgments and usage, then pause completion monitoring. No new
experiment or training is authorized by this judge-model change.

Sources: [model ID](https://help.aliyun.com/zh/model-studio/qwen3-8-max),
[API compatibility](https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope).
