# R11_new activation-precision control

- Mode: `formal`
- Engineering gate: `True`
- Counters: `{"backward_calls": 2, "full_chain_forward_calls": 54, "optimizer_steps": 0, "reader_forward_calls": 0}`
- Classification: `fp32_restores_capture_and_gradient`
- BF16 capture width: `3e-05`
- FP32 capture width: `0.05`
- FP32/BF16 widening: `1666.6666666666667`
- FP32 best negative-gradient ratio: `0.8152169766791239`

This is a fixed-target numerical precision diagnostic using oracle teacher xT.
It does not establish Picture Memory training success, Reader success, ID/OOD success, or Phase 2 eligibility.
