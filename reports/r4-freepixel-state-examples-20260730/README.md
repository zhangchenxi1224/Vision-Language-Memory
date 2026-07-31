# R4-FreePixel DreamLite 中间状态示例

## 结论与来源

原始本地历史归档没有保存逐事件 RGB state 或扩散 trajectory；其中只有训练指标与统计量，无法从标量还原像素。

本目录中的图片是在 2026-07-30 使用正式历史 checkpoint 做固定 case 回放后导出的真实模型输出，并非示意图，也不是训练当时已落盘的原始截图。

- run：`r4-freepixel-formal256-seed0-20260722-9396355`
- code commit：`9396355930565501ae4ecec8768cff535167f091`
- checkpoint：`output/endpoint.pt`
- endpoint SHA256（正式 summary 绑定值）：`1b55b8e33b1514952cb79276b5d7e8bf4863e7bbe760d9005845e6a748a71fd6`
- fixed episode：`r3-dev-semantic-000000-s0-clean`
- global seed / adapter seed：`0 / 0`
- persistent state：`float_rgb`
- state target：`none`
- canonical teacher artifacts：未加载

## Case 事件

1. Turn 0 / SET：`preferred style = formal`
2. Turn 1 / QUERY：询问当前 style
3. Turn 2 / OVERWRITE：`preferred style = modern`
4. Turn 3 / QUERY：询问更新后的 style

完整英文事件文本、各 state/latent 的统计量以及 query CE 见 `endpoint/cases.json`。

## 图片含义

每个 event 目录包含：

- `diffusion-00.png`：初始噪声 latent 的 VAE 解码；
- `diffusion-01.png` 至 `diffusion-04.png`：四个移动去噪步之后的解码；
- `state.png`：写回下一 turn 的持久 RGB state，与 `diffusion-04.png` 像素一致。

派生总览：

- `r4-endpoint-state-trajectory-contact-sheet.png`：两个事件的完整五帧轨迹；
- `r4-endpoint-state-difference.png`：SET(formal) 与 OVERWRITE(modern) 的持久状态及绝对差异。

PNG 域状态差异：

- MAE：`0.16333959`
- RMS：`0.25073752`
- max absolute delta：`0.98431373`
- 至少一个通道变化超过 `1/255` 的像素比例：`99.948%`

## 解释限制

- PNG 是对 `[0,1]` RGB state 的 8-bit 可视化，会丢失浮点精度；严谨的后续诊断应同时保存 `.pt` 或 `.npy`。
- 这些图片证明 endpoint checkpoint 在固定事件回放中产生并使用了可视 RGB state，但单个 case 不能证明图像差异就是 Reader 改善的因果来源。
- 两个最终 state 仍呈现相似的桌面图像结构，同时发生大范围像素变化；这提示记忆信号可能与生成内容、空间布局和全局像素统计纠缠，后续应结合 state-swap、差分探针和多 case 对照分析。
