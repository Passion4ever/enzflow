# enzflow 实施阶段方案

基于 Rectified Flow Matching + 非等变 DiT + atom14 表征的全原子酶生成模型（codesign: 序列+全原子结构联合生成）。

核心设计原则：
- 不用 IPA/Frame/Rigid Group/χ角重建
- 不用等变网络层，等变性完全靠 SE(3) 数据增强
- 标准 Transformer + FlashAttention，最大化效率

---

## Phase 0: 项目骨架 + 数据基础 ✅

- [x] pyproject.toml, conda 环境 (enzflow)
- [x] residue_constants.py — atom14/atom37 常量表
- [x] pdb_parser.py — PDB 解析 → atom14 张量 + 写回
- [x] 非标准残基映射 (MSE→MET 等)
- [x] 测试 22/22 通过, ruff 无错误, round-trip RMSD=0

---

## Phase 1: 数据处理 + Dataset ✅

目标：从 18W PDB → 训练可用的 DataLoader。

### Step 1.1: PDB 预处理缓存 ✅
- `scripts/preprocess_pdbs.py` — 批量 PDB → .pt 张量文件
- 183,260 个蛋白全部处理完成，零错误，12GB
- 每个 .pt 包含：coords[N,14,3], atom_mask[N,14], aatype[N], residue_index[N], ec_numbers, uniprot_id

### Step 1.2: SE(3) 数据增强 ✅
- `enzflow/data/transforms.py`
- `random_rotation_matrix()` — 均匀采样 SO(3)（QR 分解法）
- `random_se3_augmentation(coords, atom_mask)` — CA 质心归零 → 随机旋转
- 14 个测试全部通过

### Step 1.3: 几何特征 ✅
- `enzflow/data/featurizer.py`
- `rel_pos`: clamp(j-i, -32, 32) → [0, 64]，用于 embedding lookup
- `ca_dist_rbf`: CA-CA 距离 → 16 维 RBF 高斯核编码
- `ca_unit_vec`: CA_i → CA_j 单位方向向量
- **关键：训练时必须用 x_t（噪声坐标）算，不能用 x_1（真实坐标）**
- 18 个测试全部通过

### Step 1.4: EC 向量预计算 ✅
- `data/scripts/build_ec_vectors.py`
- ec_semantic.json → Qwen3-Embedding-0.6B 编码 → .pt 查找表
- 层级文本构建：level 5(完整) → level 1(仅rxn)，逐级去掉信息，rxn 始终保留
- 训练用: `weights/ec_vectors_train.pt` — 34,755 个向量 (6,951 EC × 5 levels)
- 采样用: `weights/ec_vectors_infer.pt` — 35,165 个向量 (额外 410 个 hierarchy 部分EC)

### Step 1.5: Dataset + DataLoader ✅
- `enzflow/data/dataset.py` — ProteinDataset + BucketBatchSampler
- 加载 .pt 缓存，SE(3) 增强，EC 层级 CFG dropout
- 按链长分桶 (bucketing) 减少 padding 浪费
- collate_fn 处理变长 padding
- 173,884 train / 9,376 test（Foldseek 30% SI 聚类划分）

---

## Phase 2: 模型架构 ✅

目标：模型能做前向传播，输入 batch 输出 v_theta。46 个测试全通过。

### Step 2.1: AtomEncoder ✅
- `enzflow/model/atom_encoder.py`
- atom14 坐标 + 原子位置嵌入 + 元素嵌入 + mask → 拼接 → Linear → d_atom
- 残基内注意力（3 层 Transformer，序列长度仅 14）
- 聚合：取 CA 特征 → Linear → d_token
- 输出：token_repr [B, N, d_token] + atom_repr [B, N, 14, d_atom]

### Step 2.2: PairRepresentationInit ✅
- `enzflow/model/pair_repr.py`
- 相对位置编码 + CA 距离 RBF + outer product
- 输出：pair_repr [B, N, N, d_pair]

### Step 2.3: AdaLN-Zero ✅
- `enzflow/model/adaln.py`
- cond → SiLU → Linear → (gamma, beta, alpha)
- 最后一层 zero-init → 初始时残差为恒等映射

### Step 2.4: PairformerBlock ✅
- `enzflow/model/pairformer.py`
- Token track: AdaLN → Pair-Biased Self-Attention → AdaLN → SwiGLU FFN
- Pair track: pair transition MLP（先不加三角更新）
- pair_repr → Linear → [B, N, N, n_heads] 作为 attention bias

### Step 2.5: AtomDecoder ✅
- `enzflow/model/atom_decoder.py`
- token_repr → 广播到 14 原子 + skip connection (atom_repr)
- 残基内注意力 × 3 层
- Linear → 3 维速度，虚拟原子速度置零

### Step 2.6: AllAtomFlowModel 整合 ✅
- `enzflow/model/flow_model.py`
- 时间步：正弦编码 → MLP → d_cond
- EC 条件：Qwen3 向量 → 瓶颈 MLP (1024→128→64→512)
- cond = t_embed + ec_embed（ec_embed 为零 → 无条件生成）
- 流程：AtomEncoder → PairInit → Pairformer ×12 → AtomDecoder → v_theta

### 实际超参数
```
d_token=512, d_pair=128, d_atom=128, d_cond=512
n_trunk=12, n_atom_layers=3, n_heads=8
参数量: 66.8M
```

### 过拟合验证
- B=6, N=512, bf16 → ~65GB 显存
- loss 206→25 稳定下降，管道正确

---

## Phase 3: 训练脚本 ✅

目标：全量数据正式训练，支持多卡、checkpoint、监控。

### Step 3.1: Rectified Flow Loss ✅
- `enzflow/training/flow_matching.py`
- x_0 ~ N(0,I), t ~ U(0,1), x_t = (1-t)*x_0 + t*x_1, v_target = x_1 - x_0
- loss = masked MSE(v_pred, v_target)，归一化 sum/(n_atoms*3)

### Step 3.2: 训练工具 ✅
- `enzflow/training/scheduler.py` — cosine with linear warmup
- `enzflow/training/checkpoint.py` — save/load/cleanup，保留最近 3 个

### Step 3.3: 训练入口 ✅
- `scripts/train.py` — YAML 配置 + CLI 覆盖
- HuggingFace Accelerate 管理多卡 + AMP（CUDA_VISIBLE_DEVICES 控制卡数，自动检测）
- AdamW, lr=1e-4, warmup 5K steps, cosine decay (min_lr_ratio=0.01), grad_clip=1.0, bf16
- 可选 wandb 日志 (`--wandb`)，TorchBell Telegram 通知
- 全量 464,499 AFDB 样本训练
- 每 5000 步存 checkpoint + milestone checkpoint 永久保留

### Step 3.4: 采样验证回调 ✅
- 每 ckpt_every 步自动采样 4 个蛋白（B=1 逐个，不会 OOM）
- 计算几何指标：CA-CA 距离、C-N 肽键、N-CA 键长、序列多样性
- 记录到 wandb（含 ideal 参考线），训练中实时监控结构质量

### Bug 修复记录
- **AdaLN dead gradient bug（v2 训练无效）**：PairBiasedAttention out_proj 和
  SwiGLUFFN w2 都 zero-init，与 AdaLN gate 的 zero-init 冲突，导致
  gate * output = 0 * 0 无梯度，12 个 trunk block 全部失效。
  修复：只 zero-init AdaLN gate，sublayer 用 default init。
  pretrain_v3 从头训练。

### Step 3.5: Co-design 序列预测 ✅
- 模型返回 (v_theta, seq_logits)，seq_head: Linear -> SiLU -> Linear(20)
- aatype masking: dataset 返回真实 aatype (0-19)，flow_matching 做 mask（非 motif -> 20）
- seq_loss = CE(seq_logits, aatype)，seq_loss_weight=1.0

### Step 3.6: 坐标归一化 ✅
- COORD_SCALE=9.0（训练数据中心化后坐标 std），使坐标 ~ N(0,1) 匹配噪声先验
- 自适应噪声：按每个样本 CA 方差缩放噪声，保持噪声-数据比 ~1:1
- RBF d_max 从 22A 调到 2.5 适配归一化尺度

### Step 3.7: 跨残基原子注意力 ✅
- CrossResidueBlock 替代 IntraResidueBlock（encoder + decoder）
- 滑动窗口 (i-1, i, i+1)，query 14 atoms attend to 42 atoms
- 学习 residue-offset embedding 区分相邻残基
- 使模型能直接学习肽键几何和骨架二面角

### 未实现（后续按需添加）
- [ ] 辅助 bond loss（已实现但 weight=0，开启后收益待验证）
- [ ] EMA（采样脚本再加）
- [ ] Motif-scaffolding 训练（dataset 已支持，暂未开启）

---

## Phase 4: 采样 + 评估 ✅

目标：能生成蛋白质并定量评估。

### Step 4.1: ODE 采样器 ✅
- `scripts/sample.py` — Euler 积分（默认 20 步，50 步以上无明显收益）
- batch_size x num_batches 控制生成量（按 GPU 显存调 batch_size）
- atom_mask 策略：random AA 初始化 -> 逐步 blend 到 seq_logits 预测的 hard mask
- 输出：PDB + FASTA（给 ESMFold）+ metadata.json
- 存储：outputs/（本地目录，可软链接到外部存储）

### Step 4.2: 几何质量评估 ✅
- `enzflow/evaluation/geometry.py`
- 键长/键角偏差、Ramachandran、clash 检测

### Step 4.3: Designability ✅
- `enzflow/evaluation/designability.py`
- codesign 直接出序列 -> ESMFold 折叠 -> scTM/scRMSD（可选，需 fair-esm）

### Step 4.4: 多样性 ✅
- `enzflow/evaluation/diversity.py`
- 生成 N 个结构的 pairwise TM-score 分布

### 训练实验记录

**pt_v7 (2026-03-14 ~ 03-17, 最终版)**
- 配置: 67M params, batch=6, lr=1e-4, warmup=5K, 2卡DDP, seq_loss_weight=1.0
- 数据: 464,499 AFDB 样本，1M 步（~25.8 epoch），耗时 77.3h
- checkpoint: `checkpoints/pt_v7_0314_012202/`

验证回调趋势（训练中 B=1, L=100, 10 samples）:

| 步数 | Bond MAE | Angle MAE | Rama | Clash | AA |
|------|----------|-----------|------|-------|----|
| 5K | 0.486A | 35.9 deg | 68% | 0.0014 | 15/20 |
| 50K | 0.195A | 12.3 deg | 97% | 0.0002 | 20/20 |
| 150K | 0.112A | 7.3 deg | 98% | 0.0003 | 20/20 |
| 350K | 0.061A | 4.3 deg | 99% | 0.0001 | 20/20 |
| **500K (最佳)** | **0.053A** | **3.6 deg** | **99%** | **0.0001** | **20/20** |
| 1M | 0.079A | 5.3 deg | 98% | 0.0001 | 20/20 |

正式采样评估（500K 步，100 samples，L=100/200/300/500/800）:

| 指标 | 值 | 目标 | 状态 |
|------|----|------|------|
| Bond MAE | 0.068A | < 0.05A | 接近 |
| Bond within 0.05A | 50% | - | - |
| Angle MAE | 4.58 deg | < 5 deg | **达标** |
| Angle within 5 deg | 66% | - | - |
| Ramachandran | 98.3% | > 90% | **远超** |
| Clash ratio | 0.004% | < 5% | **远超** |
| Diversity (TM) | 0.248 | 低 = 好 | 良好 |
| AA types | 20/20 | 20/20 | **达标** |
| scTM (designability) | 未测 | > 0.5 @60% | 待测 |

采样步数对比（500K checkpoint，100 samples）:

| Steps | Bond MAE | Angle MAE | 备注 |
|-------|----------|-----------|------|
| 10 | 0.118A | 7.48 deg | 太少 |
| 50 | 0.069A | 4.73 deg | 够用 |
| 100 | 0.068A | 4.58 deg | 边际收益小 |

观察:
- 最佳验证指标在 ~400K-550K 步，之后震荡退化
- 500K 步为当前最佳 checkpoint
- Designability (ESMFold scTM) 尚未测试

---

## Phase 5: EC 条件采样 + Motif-Scaffolding 采样 ⬜

目标：支持层级 EC 条件引导采样和 motif-scaffolding 采样。

### Step 5.1: 层级 CFG 采样
- 完整 4 级 EC → 3 次前向：v = v_uncond + w_rxn*(v_rxn - v_uncond) + w_full*(v_full - v_rxn)
- 部分 EC（无 rxn）→ 2 次前向：v = v_uncond + w*(v_cond - v_uncond)

### Step 5.2: Motif-Scaffolding 采样
- scaffold 从噪声开始，motif 用真实坐标的 t-插值
- 每步只更新 scaffold 区域

### 验证标准
- 3 个不同 EC 大类各生成 50 个结构，t-SNE 应按 EC 聚类
- motif RMSD < 1.0Å，scaffold scTM > 0.5

---

## 数据策略（待定）

当前训练数据：18W AFDB 酶（7,702 个 30% SI families）

潜在扩展方案：
- Foldseek AFDB 聚类子集：230 万 cluster，过滤后 54.6W 代表（50-512 aa, pLDDT>=80, 非dark）
- 方案 B（两阶段）：先用 Foldseek 子集预训练学通用几何，再用 18W 酶微调加 EC 条件
- 决策：先用 18W 酶训练到 Phase 4 看效果，如果几何质量不够再加通用数据
