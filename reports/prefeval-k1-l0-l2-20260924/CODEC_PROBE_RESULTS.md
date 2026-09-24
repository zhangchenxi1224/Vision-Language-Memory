# 固定首例：VAE 往返保留了 B 的正确读取

2026-09-24。按[运行前方案](CODEC_PROBE_PROTOCOL.md)完成固定首例的 10 张原图、
10 张一次往返图读取，均为 T1 官方 MCQ。正确选项为 D。

| 输入 | A 原图 → 往返图 | B 原图 → 往返图 |
|---|---|---|
| 教师目标图 | C → C | D → D |
| write2048 初写，chain 0 | C → C | D → D |
| write2048 初写，chain 1 | C → C | D → D |
| retain2048 初写，chain 0 | A → A | D → D |
| retain2048 初写，chain 1 | C → C | D → D |

B 的五个配对全部保持正确；A 起点均错，不能据此衡量信息保持成功。
10 次原图重读与此前记录的图像哈希、问题、选项、原始输出 token 全部一致。
20 条记录无解析失败和输出截断。像素确有变化，B 的 [0,1] 像素 MSE 约 0.0000985—0.0001039，
A 约 0.000647—0.000749；不同目标图分布下不将跨组像素误差直接解释为能力优劣。

本次调用冻结官方 pipeline 的原构造函数及 `prepare_image_latents`，
只加载 VAE，不加载不必要的 Writer 组件；实际 float32 编码后经既有解码及 uint8 PNG 落盘，
重新读取后才交给 Reader，没有用浮点图片绕过 PNG。

结合[真实 Writer 首轮干扰失败](FIRST_POSTRETAIN_DIAGNOSTIC.md)，
这一案例不支持“仅一次 VAE 往返就足以导致该失败”的解释，定位重点转向 Writer 保持映射。
它不能证明 VAE 在所有偏好、多次往返上都无影响，也不能说明具体哪条条件路径未被模型有效利用。
下一轮[仅用学生初写图学习保持](INITIAL_SOURCE_RETAIN_PROTOCOL.md)，检验深层失败前缀作为训练输入的影响。

- [原始 20 条读回](evidence/codec-probe/first-case-codec-probe/readback.jsonl)
- [输入、官方源码和执行提交绑定](evidence/codec-probe/first-case-codec-probe/manifest.json)
- [往返像素误差与 PNG 哈希](evidence/codec-probe/first-case-codec-probe/codec.jsonl)
- [重复读取与 parser 独立核验](evidence/codec-probe/first-case-codec-probe/local-verification.json)
