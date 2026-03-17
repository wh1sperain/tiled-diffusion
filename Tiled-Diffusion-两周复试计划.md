# Tiled Diffusion 两周复试项目计划

## 1. 项目目标

两周内完成一个适合研究生复试展示的 Tiled Diffusion 项目，不停留在简单复现，而是形成一套完整的“复现 + 分析 + 小改进 + 面试表达”成果。

项目最终定位为：

> 围绕扩散模型生成无缝可拼贴图像（Tileable Image）的问题，复现 CVPR 2025 的 Tiled Diffusion 方法。该方法通过在 latent 空间去噪过程中对相邻 tile 的边界区域施加一致性约束，实现多张图像的无缝拼接。我在复现基础上，对上下文宽度（max_width）和一致性约束宽度（max_replica_width）等关键参数进行消融实验，并尝试改进 padding 区域的融合策略以提升边界连续性。

## 2. 最终交付物

两周结束时，至少准备以下 6 项成果：

1. 一个能够稳定运行的 Tiled Diffusion demo，支持 self-tiling、one-to-one、many-to-many 等拼接模式。
2. 一组固定测试样例（固定 prompt、seed），覆盖纹理、风景、建筑等场景。
3. 一张实验对比表，包含拼接无缝性、生成时间、CLIP 相似度等指标。
4. 一个小改进，对 padding 区域的边界融合策略进行优化。
5. 一套 8 到 10 页的复试 PPT。
6. 一段 3 分钟项目介绍稿和一套常见追问回答。

## 3. 项目主线

复试时不要把项目讲成“我跑通了一个开源项目”，而应讲成：

> 我复现了 CVPR 2025 的 Tiled Diffusion，关注的是如何让扩散模型生成可无缝拼贴的图像。传统方法依赖手动拼接或后处理，而 Tiled Diffusion 在扩散去噪过程中，通过在 latent 空间为每个 tile 添加 padding 区域、并在每步去噪时交换相邻 tile 的边界信息，使多张图像在生成过程中自然趋向无缝。我在复现基础上，对上下文宽度 max_width 和一致性约束宽度 max_replica_width 做了消融实验，并比较了不同 padding 区域融合策略对拼接质量的影响。

## 4. 两周执行安排

### 第 1 阶段：跑通与理解（Day 1 - Day 3）

#### Day 1：环境搭建与首次运行

目标：先把项目跑通。

任务：

1. 搭建 conda 环境，安装 requirements.txt 依赖，记录 CUDA、PyTorch、显卡版本。
2. 运行 run.py，验证 self-tiling 模式可用（生成一张可与自身无缝拼接的图像）。
3. 尝试不同拼接模式（self-tiling、one-to-one、many-to-many），确认都能跑通。
4. 尝试 img2img 模式（设置 source_image）。

当天产出：

1. 一份环境配置记录。
2. self-tiling、one-to-one、many-to-many 各一张输出图，以及对应的拼接预览图。
3. 一份简短笔记，写清 prompt、side_id/side_dir 配置、主要参数和输出效果。

#### Day 2：理解方法与代码结构

目标：搞清楚方法机制，而不是只会运行。

任务：

1. 梳理代码文件职责：latent_class.py（tile 状态封装）、latent_handler.py（核心算法：tile/similarity/random_padding 三大约束）、model.py（主流程：加载模型、去噪循环、VAE 解码）、utils.py（图连接构建、tensor 切片）、config.py（旋转矩阵、方向常量）。
2. 理解核心流程：初始化带 padding 的 latent → 每步去噪时执行 similarity_constraint + tile 边界交换 → UNet 去噪 → 最终 VAE decode 并裁切 padding。
3. 找到关键参数入口：max_width（上下文宽度/padding 大小）、max_replica_width（similarity 约束宽度）、inference_steps、cfg_scale、seed、side_id/side_dir（边连接关系）。
4. 理解 generate_graph_groups() 如何根据 side_id 和 side_dir 构建边匹配图。
5. 理解旋转矩阵（SIMILARITY_ROTATION_MATRIX、TILING_ROTATION_MATRIX）的作用：保证不同方向的边在复制时正确对齐。

当天产出：

1. 一张自己画的方法流程图，包含 latent padding、边界约束、去噪循环、裁切等步骤。
2. 一段 1 分钟口头解释稿，能清晰说明方法如何在去噪过程中实现无缝拼接。

#### Day 3：固定测试方案

目标：建立后续实验的统一标准。

任务：

1. 选择 3 到 5 个测试 prompt，覆盖不同场景。
2. 固定随机种子（seed），保证结果可对比。
3. 固定输出分辨率为 512x512（SD 1.5 的标准分辨率）。
4. 确定主要测试的拼接模式：优先用 self-tiling（side_id=[1,1,2,2]），因为最能体现无缝效果。
5. 制定拼接质量评价方式：将生成图与自身拼接后观察边界。

建议测试场景（尽量选择 tileable 属性明显的）：

1. 砖墙、石材等纹理图，天然适合 tiling，便于观察纹理过渡。
2. 风景图，观察大尺度结构在拼接处的连续性。
3. 抽象或几何图案，观察图案周期性。
4. 自然元素（木头、草地），观察自然随机纹理的无缝效果。

当天产出：

1. 一套固定测试样例（prompt + seed + side_id/side_dir 配置）。
2. 一份实验记录模板。

### 第 2 阶段：核心实验（Day 4 - Day 7）

#### 总体目标

通过参数消融实验证明你对方法的理解，而不是只给出单张生成图。

优先分析两个核心参数：

1. max_width：上下文宽度，即 latent 空间中 padding 区域的大小，决定相邻 tile 能“看到”多少邻居信息。
2. max_replica_width：similarity 约束宽度，决定多对多场景下 tile 内侧一致性约束的范围。

建议实验配置：

1. max_width：4、8、12、16、32
2. max_replica_width：1、3、5、8

如果时间有限，不要做全组合。建议先固定 max_replica_width 比较 max_width，再固定一个较优 max_width 比较 max_replica_width。

#### Day 4：max_width 消融

任务：

1. 固定 prompt、seed、inference_steps、cfg_scale 和输出分辨率。
2. 固定 max_replica_width=3，分别设置 max_width=4、8、12、16、32。
3. 对每组结果，将生成图与自身拼接形成 2x2 预览图，观察拼接处。
4. 记录每组生成时间与显存占用。

观察重点：

1. max_width 太小时，拼接处是否存在明显纹理不连续。
2. max_width 增大后，拼接处过渡是否更自然。
3. max_width 太大时，是否反而导致图像中心区域质量下降（因为 UNet 处理的 latent 尺寸增大）。
4. 生成时间和显存的变化趋势。

#### Day 5：max_replica_width 消融

任务：

1. 固定其他参数，固定 max_width 为 Day 4 得到的较优值。
2. 分别设置 max_replica_width=1、3、5、8。
3. 重点在 many-to-many 场景下测试（多个 tile 需要互相拼接）。
4. 对边界区域进行局部放大观察。

观察重点：

1. max_replica_width 对 many-to-many 场景中多个 tile 边界一致性的影响。
2. 增大 max_replica_width 是否能显著提升不同 tile 之间的对称性。
3. 生成速度是否明显下降。

#### Day 6：补充实验与初步结论

任务：

1. 对比不同拼接模式（self-tiling vs one-to-one vs many-to-many）的效果差异。
2. 尝试 img2img 模式，观察不同 strength 对原图保留度和拼接质量的影响。
3. 观察不同 scheduler（ddpm vs ddim）的效果差异。
4. 写出 3 到 5 条初步结论。

可形成的结论方向：

1. max_width 太小时，相邻 tile 交换的边界信息不足，接缝明显。
2. max_width 增大能有效提升拼接无缝性，但 latent 尺寸增大导致显存和时间开销上升。
3. max_replica_width 对 many-to-many 场景的边界对称性有明显影响。
4. img2img 模式下，strength 决定原图保留与拼接质量的权衡。

#### Day 7：整理实验结果

任务：

1. 整理图表与参数记录。
2. 制作拼接预览图和边界局部放大图的对比拼图。
3. 形成一页“参数影响分析”材料。
4. 可以使用 gif_creator.py 生成滑动 GIF 来直观展示无缝拼接效果。

当天产出：

1. 一张实验对比表（max_width / max_replica_width / 时间 / 显存 / 拼接质量）。
2. 一组可视化对比图和无缝滑动 GIF。
3. 3 到 5 条清晰结论。

### 第 3 阶段：做一个小改进（Day 8 - Day 10）

#### 总体目标

在复现基础上加入一项你自己的工作，优先选择“小而合理、易于解释、容易验证”的改进。

最推荐的方向：padding 区域边界融合策略改进。

当前代码的基线做法：在 latent_handler.py 的 tile() 方法中，将相邻 tile 的边界切片直接赋值到当前 tile 的 padding 区域（硬覆盖）。这种方式可能导致 padding 区域与 tile 内部之间出现突变。

建议对比以下 3 种方式：

1. 直接赋值（当前基线）：源 tile 边界切片直接覆盖目标 tile 的 padding 区域。
2. 平均融合：padding 区域 = 0.5 * 原值 + 0.5 * 源切片。
3. 加权融合：在 padding 区域内构建从 0 到 1 的线性渐变权重，让距离 tile 内部越近的位置保留更多原始信息，距离越远的位置更多采用邻居切片。

加权融合的思路可以表述为：

> 在 padding 区域内构建线性渐变权重，使 tile 内部与相邻 tile 边界之间实现平滑过渡，避免硬覆盖带来的突变。

#### Day 8：明确改进方案

任务：

1. 仔细阅读 latent_handler.py 中 tile() 方法的赋值逻辑，理解当前硬覆盖的位置和形状。
2. 画出 padding 区域融合示意图，标注权重分布。
3. 确定要实现和比较的融合策略，写清代码修改点。

#### Day 9：运行改进实验

任务：

1. 在固定 prompt 和参数设置下运行改进前后实验。
2. 对比拼接预览图和边界局部图。
3. 记录改进前后的时间代价和视觉效果。
4. 可以用 evaluator.py 中的 CLIP 相似度或 LPIPS 做定量对比。

#### Day 10：总结改进效果

任务：

1. 统计改进前后的差异。
2. 不夸大结论，只说“在部分样例上缓解边界突变”。
3. 同时总结其局限性。

建议表述：

> 加权融合策略在 padding 边界区域提供了更平滑的过渡，在纹理类图像上能有效减少拼接处的突变感。但它不能解决全局语义一致性问题，例如拼接后图像的整体结构不连贯。这说明融合策略只能缓解边界层面的问题，更深层的一致性需要依赖更强的全局约束机制。

### 第 4 阶段：整理复试材料（Day 11 - Day 12）

目标：把实验内容转化为可以展示和讲解的成果。

建议 PPT 结构：

1. 研究背景：图像无缝拼贴在纹理创建、游戏资产、360° 全景合成等领域的重要性，传统方法依赖手动制作或后处理。
2. 问题定义：如何让扩散模型生成的多张图像在指定边自然无缝拼接，支持 self-tiling、one-to-one、many-to-many 等模式。
3. 方法流程：带 padding 的 latent 初始化 → 每步去噪时进行边界约束（similarity constraint + tiling） → UNet 去噪 → VAE decode 并裁切。
4. 复现过程：环境、模型（SD 1.5）、测试设置、固定 prompt 和 seed。
5. 参数实验：max_width 和 max_replica_width 的消融分析。
6. 结果展示：拼接预览图、边界放大图、无缝滑动 GIF。
7. 我的工作：参数消融、padding 区域融合策略改进。
8. 结论与局限：优势、问题与后续优化方向。

### 第 5 阶段：面试训练（Day 13 - Day 14）

目标：把“做过项目”转化为“讲得清楚项目”。

需要准备三套表达：

1. 30 秒版本：回答“你做了什么项目”。
2. 2 分钟版本：回答“你具体做了什么”。
3. 追问回答版本：应对机制、参数、改进与局限性问题。

30 秒版本示例：

> 我复现了 CVPR 2025 的 Tiled Diffusion，关注的是如何让扩散模型生成可无缝拼贴的图像。这个方法通过在 latent 空间为每个 tile 添加 padding 区域，并在去噪过程中交换相邻 tile 的边界信息，让多张图像在生成时自然趋向无缝。我在复现基础上，对上下文宽度 max_width 和一致性约束宽度 max_replica_width 做了消融实验，并尝试了加权融合改进 padding 区域的边界过渡效果。

## 5. 高频面试问题与回答思路

### 1. Tiled Diffusion 解决什么问题？

它解决的是如何让扩散模型生成可无缝拼贴的图像（tileable image）。传统做法需要手动制作或后处理，而 Tiled Diffusion 在生成过程中自动保证指定边的无缝性，支持 self-tiling、one-to-one、many-to-many 多种拼接模式，可应用于纹理创建、游戏资产、360° 全景合成等领域。

### 2. Tiled Diffusion 的核心机制是什么？

核心是在 latent 空间的去噪过程中对相邻 tile 的边界施加约束。具体做法是：为每个 tile 的 latent 添加 padding 区域（大小由 max_width 控制），然后在每步去噪时，将相邻 tile 的边界切片复制到当前 tile 的 padding 区域。这样 UNet 在去噪时能“看到”邻居的边界信息，从而生成与邻居一致的过渡。最终 VAE decode 后裁掉 padding 区域，得到边界自然无缝的图像。

### 3. 代码中有哪些关键约束？

有三个约束，均在 latent_handler.py 中实现：

1. **Tiling 约束**：tile() 方法，将匹配 tile 的边界切片复制到当前 tile 的 padding 区域，这是保证无缝拼接的核心操作。
2. **Similarity 约束**：apply_similarity_constraint() 方法，在 many-to-many 场景下，让共享同一连接模式的多个 tile 的内侧边界区域保持一致。
3. **Random Padding 约束**：apply_random_padding_constraint() 方法，仅在去噪结束后调用一次，用于最终对齐。

### 4. max_width 和 max_replica_width 分别控制什么？

max_width 是 padding 区域的宽度（latent 空间单位，对应像素 8×max_width），决定了相邻 tile 能交换多少边界信息。max_width 越大，UNet 看到的邻居上下文越多，拼接越自然，但显存和计算开销也越大。max_replica_width 是 similarity 约束的宽度，主要影响 many-to-many 场景下多个 tile 边界的一致性。

### 5. 你的工作体现在哪里？

可以从四点回答：

1. 我完成了可复现的环境搭建和流程梳理，理解了 latent 空间边界约束的核心机制。
2. 我设计了固定测试方案，覆盖多种拼接模式和纹理场景。
3. 我做了 max_width 和 max_replica_width 的参数消融，分析了拼接质量与计算开销之间的权衡。
4. 我尝试了将 padding 区域的硬覆盖改为加权融合，并总结了其有效性和局限性。

### 6. 你的改进为什么合理？

因为当前代码在 tile() 方法中直接用邻居的边界切片覆盖 padding 区域，这种硬赋值可能在 padding 与 tile 内部之间形成突变。通过加权融合，让 padding 区域从 tile 内部到外侧逐渐从原始值过渡到邻居切片，可以减少这种突变，让 UNet 在去噪时获得更平滑的输入。

### 7. 这个方法的局限性是什么？

1. padding 区域的大小有限，当图像需要长程全局结构一致性时，仅靠边界交换无法保证。
2. 每步去噪都要交换边界信息，当 tile 数量增多时计算开销会上升。
3. 方法主要针对纹理类图像效果最好，对于具有明确全局语义结构的图像（如人物、建筑），拼接后可能存在语义不连贯。
4. 旋转对齐依赖预定义的旋转矩阵，灵活性有限。

### 8. side_id 和 side_dir 如何工作？

side_id 是一个长度为 4 的列表，对应 [Right, Left, Up, Down]，每个值是连接组 ID。两条边如果有相同的 side_id 且方向互补（cw 对 ccw），就会被匹配为可拼接的一对。例如 side_id=[1,1,None,None], side_dir=['cw','ccw',None,None] 表示左右两边可以互连，上下不连接。generate_graph_groups() 函数会根据这些信息自动构建边匹配图。

### 9. 为什么需要旋转矩阵？

因为当一个 tile 的右边要与另一个 tile 的左边拼接时，切片方向不同，特别是当上下边和左右边交叉拼接时，需要对 tensor 做旋转才能正确对齐。TILING_ROTATION_MATRIX 和 SIMILARITY_ROTATION_MATRIX 预定义了每种方向组合所需的 rot90 次数。

## 6. 实验记录模板

每次实验建议记录以下内容：

1. 模型版本（如 SD 1.5 / SDXL / SD 3）。
2. prompt 和 negative_prompt。
3. seed。
4. 输出分辨率（height x width）。
5. max_width（上下文宽度 / padding 大小）。
6. max_replica_width（similarity 约束宽度）。
7. inference_steps（去噪步数）。
8. cfg_scale。
9. scheduler（ddpm / ddim / euler）。
10. 拼接模式（self-tiling / one-to-one / many-to-many）和 side_id/side_dir 配置。
11. strength（img2img 模式下）。
12. 生成时间。
13. 最大显存占用。
14. 主观观察，重点记录拼接处无缝性、纹理连续性、整体一致性。

## 7. 最小可行版本

如果时间不够，至少保证完成以下内容：

1. 跑通 Tiled Diffusion，验证 self-tiling、one-to-one、many-to-many 三种模式。
2. 做 max_width 和 max_replica_width 两组消融实验。
3. 做拼接处边界区域可视化对比（将生成图与自身拼接展示）。
4. 尝试一个简单的加权融合方案（修改 latent_handler.py 中 tile() 方法的赋值逻辑）。
5. 准备 8 页左右 PPT 和 3 分钟讲稿。

## 8. 注意事项

为了保证两周内顺利完成，建议主动避开以下风险：

1. 不要尝试从头训练大模型。
2. 不要追求过度复杂的创新。
3. 不要在没有合适参考标准时硬做复杂图像指标。
4. 不要频繁更换模型、环境或测试设置。
5. 不要把项目做成单纯的安装教程。

项目真正要展示的是：

1. 你理解了什么。
2. 你分析了什么。
3. 你验证了什么。
4. 你能否清楚表达方法的机制、权衡与局限。