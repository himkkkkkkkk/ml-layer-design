// Qwen3-4B 中间层语义空间分析报告
// 编译: typst compile report.typ

#set document(
  title: "Qwen3-4B 中间层语义空间分析报告",
  author: "Layer Analysis Team",
  date: datetime.today(),
)

#set page(
  paper: "a4",
  margin: (x: 2.5cm, y: 2cm),
)

// #set text(font: ("Noto Sans CJK SC", "Noto Sans"), size: 11pt, lang: "zh")
#set heading(numbering: "1.")
#set figure(gap: 8pt)
#set table(stroke: 0.5pt, inset: 8pt)
#set par(leading: 0.6em, spacing: 0.8em)

// ── 概述 ──
= 概述

本文档记录了在 Qwen3-4B-Instruct-2507 模型上进行的五组实验，旨在验证以下假设：

#quote(block: true)[
  Transformer 模型的中间层存在一个 *语言无关的语义空间（interlingua）*，
  该空间编码的是"含义"而非"表面形式"。
]

实验设计参考了 David Ng 的博客文章 _\"LLM Neuroanatomy III: Do LLMs Break the Sapir-Whorf Hypothesis?\"_。

#pagebreak

// ══════════════════════════════════════════════════════════════
// 实验一
// ══════════════════════════════════════════════════════════════
= 实验一：多语言 Sapir-Whorf 测试

== 方法

- *数据*：8 个常识事实 × 5 种语言（英/中/德/法/日）× 2 种句式（陈述/疑问）= 80 条 prompt
- *指标*：逐层计算余弦相似度
  - #text(fill: red)[RED]：同一事实，不同语言（如 EN"sun rises" $↔$ ZH"太阳升起"）
  - #text(fill: green)[GREEN]：同一语言，不同事实（如 EN"sun rises" $↔$ EN"water boils"）
  - #text(fill: blue)[BLUE]：不同事实，不同语言（基线）

== 核心曲线

#figure(
  image("results/sapir_whorf/sw_main_statements.png", width: 100%),
  caption: [RED/GREEN/BLUE 余弦相似度曲线（陈述句）],
)

#figure(
  image("results/sapir_whorf/sw_centered_statements.png", width: 100%),
  caption: [中心化余弦相似度（减去 BLUE 基线）],
)

== 结果

#table(
  columns: (auto, auto),
  [*层范围*], [*主导因素*],
  [L0–L5], [GREEN > RED → 语言占主导],
  [L6–L22], [RED > GREEN → 语义占主导（interlingua 出现！）],
  [L23–L35], [GREEN > RED → 语言回归（输出准备）],
)

- RED > GREEN 的层数：17/37（46%）
- 峰值 RED-GREEN 差距：约 +0.21

== 聚类图

#figure(
  image("results/sapir_whorf/pca_topic_all_s.png", width: 100%),
  caption: [PCA 按话题聚类——陈述句，全部 37 层],
)

#figure(
  image("results/sapir_whorf/pca_topic_all_q.png", width: 100%),
  caption: [PCA 按话题聚类——疑问句，全部 37 层],
)

== 结论

即使是在 4B 规模的指令模型上，也存在三层结构：*输入层处理语言 → 中间层编码语义 → 输出层回归语言*。与博客中 7B–72B 指令模型相比，效应较弱但模式一致。说明 interlingua 是预训练的自然涌现属性，指令微调仅起到放大作用。

#pagebreak

// ══════════════════════════════════════════════════════════════
// 实验二
// ══════════════════════════════════════════════════════════════
= 实验二：句法变换测试

== 方法

- *数据*：8 个事实 × 6 种句法形式 = 48 条 prompt，仅英语
  - plain / inverted / cleft / emphatic / impersonal / topicalized
- *可视化*：PCA 图中 *颜色 = 话题*，*形状 = 句法结构*

== 核心图表

#figure(
  image("results/syn_test/pca_combined_all.png", width: 100%),
  caption: [PCA：颜色 = 话题，形状 = 句法结构——全部 37 层],
)

#figure(
  image("results/syn_test/syn_rgb_curves.png", width: 100%),
  caption: [句法 RGB 曲线：RED = 同话题不同句法，GREEN = 同句法不同话题],
)

== 结果

- 全层平均相似度：句法变化 *0.81*（最高 = 最不敏感）
- L13–L18：句法变化 *0.80*

== 结论

句法结构对模型的中间层表示 *几乎没有影响*。同一事实无论用什么句法表达，向量都高度聚类。语法形式已被模型压缩。

#pagebreak

// ══════════════════════════════════════════════════════════════
// 实验三
// ══════════════════════════════════════════════════════════════
= 实验三：语用模式测试

== 方法

- *数据*：8 个事实 × 3 种语用模式（陈述/疑问/命令）× 4 种变体 = 96 条 prompt
- *可视化*：PCA 图中 *颜色 = 话题*，*形状 = 语用模式*

== 核心图表

#figure(
  image("results/prag_test/pca_combined_all.png", width: 100%),
  caption: [PCA：颜色 = 话题，形状 = 语用模式——全部 37 层],
)

#figure(
  image("results/prag_test/prag_rgb_curves.png", width: 100%),
  caption: [语用 RGB 曲线：RED = 同话题不同模式，GREEN = 同模式不同话题],
)

#figure(
  image("results/prag_test/prag_per_mode.png", width: 100%),
  caption: [各模式内话题聚类对比：陈述 vs 疑问 vs 命令],
)

== 结果

#table(
  columns: (auto, auto, auto),
  [*对比维度*], [*L13–L18 相似度*], [*全层平均*],
  [句法变化], [0.80], [0.81],
  [语用模式变化], [0.58], [0.58],
  [语言变化], [0.68], [0.58],
)

== 结论

语用模式造成的表示差异 *远大于* 句法变化（0.58 vs 0.81），且与语言差异（0.58）大小相当。"提问"vs"陈述"vs"命令"是一种更深层的语义信号，模型不会将其与句法形式一同压缩。疑问句的特殊性不在于语言形式，而在于其承载的 *语用行动*（请求回答）。

#pagebreak

// ══════════════════════════════════════════════════════════════
// 实验四
// ══════════════════════════════════════════════════════════════
= 实验四：迭代生成测试

== 方法

- 初始 prompt → 生成 → 用生成结果作为新 prompt → 重复 20 次
- 追踪各层连续迭代间的余弦相似度

== 核心图表

#figure(
  image("results/iterate/consecutive_sim.png", width: 100%),
  caption: [连续迭代相似度——各层对比],
)

#figure(
  image("results/iterate/drift_from_original.png", width: 100%),
  caption: [与原始 prompt 的偏离度——各层对比],
)

== 结果

#table(
  columns: (auto, auto, auto),
  [*层*], [*连续相似度*], [*含义*],
  [emb], [0.03], [词元层面完全发散],
  [L0–L5], [0.37→0.50], [编码阶段，逐渐稳定],
  [L6–L23], [0.45–0.57], [活跃处理，差异最大],
  [L24–L34], [0.57→*0.79*], [*收敛到稳定主旨*],
  [L35], [0.46], [输出投射，强制选择 token],
)

== 结论

最稳定的层是 *L30–L34（后期层）*，而非中间层。中间层（L6–L23）变化最大——它们在做"工作"（处理具体措辞和细节）。后期层将细节压缩，回归到稳定的核心语义。

#pagebreak

// ══════════════════════════════════════════════════════════════
// 实验五
// ══════════════════════════════════════════════════════════════
= 实验五：多话题迭代聚类测试

== 方法

- 8 个话题 × 5 次迭代 = 40 个向量
- PCA 降维，颜色标记话题

== 核心图表

#figure(
  image("results/iterate_multi/pca_topic_all.png", width: 100%),
  caption: [PCA 按话题聚类——全部 37 层，8 话题 × 5 迭代],
)

#figure(
  image("results/iterate_multi/within_vs_cross.png", width: 100%),
  caption: [话题内 vs 跨话题相似度——各层对比],
)

== 结果

#table(
  columns: (auto, auto, auto, auto),
  [*层*], [*话题内相似度*], [*跨话题相似度*], [*差距*],
  [L10], [0.51], [0.29], [0.21],
  [L11], [0.53], [0.30], [0.23],
  [L12], [0.53], [0.31], [0.22],
  [L30], [0.71], [0.58], [0.13],
  [L34], [0.83], [0.76], [0.07],
)

== 结论

中层（L10–L12）话题内/跨话题 *差距最大*（0.22），说明中层最能区分不同话题。后期层话题内相似度最高（0.83）但跨话题相似度也很高（0.76），差距缩小——后期层压缩了话题间的差异，趋向普适表示。

#pagebreak

// ══════════════════════════════════════════════════════════════
// 综合结论
// ══════════════════════════════════════════════════════════════
= 综合结论

== 模型的层状处理管线

#set par(leading: 0.4em, spacing: 0.4em)
#text(size: 10pt)[
  ```
  L0–L5:    输入编码 —— 词元 → 初步表示
  L6–L12:   语言处理 —— 剥离语言特定特征
  L13–L18:  语义提取 —— 语言被擦除，话题开始聚类（SW 效应）
  L19–L29:  细节加工 —— 处理具体措辞和阐述
  L30–L34:  主旨压缩 —— 压缩细节，回归稳定核心语义（迭代稳定性最高）
  L35:      输出投射 —— 强制选择下一个 token
  ```
]
#set par(leading: 0.6em, spacing: 0.8em)

== 影响语义空间的三个层次

#table(
  columns: (auto, auto, auto),
  [*因素*], [*相似度*], [*影响程度*],
  [句法结构（plain vs cleft vs inverted...）], [0.80], [几乎无影响],
  [语用模式（陈述 vs 疑问 vs 命令）], [0.58], [显著影响],
  [语言（英语 vs 中文 vs 德语...）], [0.58], [显著影响],
)

== 与博客的差异

博客使用的 7B–72B 指令模型显示约 25 层 RED > GREEN，而我们的 4B 指令模型仅 17 层。差距来自模型规模——interlingua 随规模增长而增强，4B 模型已展现该结构但不够强。

== 核心发现

#quote(block: true)[
  *语言不影响思想*（Sapir-Whorf 假说在 LLM 中被推翻）。

  但 *语用意图*（问/说/命令）是一种比语言本身更深刻的语义信号。

  模型在 *后期层而非中间层* 达到最大表示稳定性——"意义空间"是一个压缩过程的结果，而非一个静态的存储层。
]
