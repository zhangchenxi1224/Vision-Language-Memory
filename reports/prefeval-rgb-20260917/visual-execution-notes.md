# Fixed visual panel execution

Plan: `prefeval-rgb-visual-recovery-sentinel-03`; policy digest
`d2e1e58f99dfd227808425fc4e2c8d79aee4444c313d971803ac4e6e7cdfba2f`.
The original v2 combined gate remains failed. This policy prospectively allocates
the visual pilot using the already passing full-value recovery references;
all auxiliary questions remain unchanged and are still measured.

The designated four-H200 notebook executed all 124 baseline writes at `3f3bf10`.
Independent raw-token/PNG-chain verification found exactly 124 expected writes,
no missing cells, singleton recovery 0/192, singleton MCQ 33/96, recurrent recovery
16/176, zero of 28 complete recurrent states, and zero of four complete episodes.
Each of the four recurrent episodes contains seven native writes. Correctly
answering an absent slot alone is insufficient to pass selective clear.

Three technical startup failures are preserved, none spent a latent optimizer step:

1. At `338650e`, the allocation verifier compared Windows backslash file labels
   to Linux slash labels. The fix at `3f3bf10` normalizes only path separators;
   file-byte digests, raw scores, queries and reference counts remain identical.
2. The initial teacher startup at `3f3bf10` failed its exact gray encoding check.
   A strict CUDA diagnostic found identical gray pixel values but different
   NCHW versus channels-last strides. The helper/native latent maximum difference
   was 1.3947486877441406e-05; helper versus contiguous-native was exactly zero,
   and official versus direct encoding with native strides was exactly zero.
   `292d5a3` therefore uses the official PIL-preprocessed gray128 encoding directly,
   preserving its native strides. This is the registered gray initializer, with
   no alternate image, noise seed, coordinate scaling or source origin.
3. That startup passed the native equality check but failed to serialize a
   `Path` in the VAE configuration receipt. `191d24a` passes the same model path
   as a string to the loader, making the unchanged snapshot identity serializable.

The actual 40-target optimization began at `191d24a`, in
`visual-recovery-v1-run/sentinel-native-gray-r2`, with 256 steps/state, three
registered recovery forms, frozen VAE/Reader, and no shared Writer updates.
The first measured finite, nonzero gradients are part of those registered steps.
No warm-up optimizer run or outcome-dependent extra update was used.

The first 49 focused tests passed before dispatch. After the native-layout fix,
13 relevant visual-policy/Base tests passed. Baseline and teacher execution
commits are intentionally recorded separately; no baseline write was repeated.

Teacher coverage and final allocation decision are pending completion of the
whole fixed panel. Partial successful targets are not a usable model result.

Both baseline archive byte hashes were independently matched after transfer:
`7ca4ba3461262e1782cf15279c857f4b193c55805722b4beceae3ad2b6eacd4e`
(shard 0) and `94417bcb2c95184727e72f628f9ab0fce712996f35b0244d0d4e14bfc5380541`
(shards 1–3). Full PNG-chain, noise-seed and raw-score verification passed in
the original Linux/Torch 2.7 execution environment. A repeat under local
Windows/Torch 2.11 stopped at noise regeneration: the same seed did not produce
the same noise-byte hash across those runtimes. The original-runtime verification
and identical archived bytes remain the evidence; cross-runtime bitwise RNG
portability is not claimed. No score or hash assertion was relaxed.
