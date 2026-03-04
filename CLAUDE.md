# enzflow

始终用中文回复。

## 这是什么

全原子酶生成模型。给定 EC 号，从噪声生成酶的全原子 3D 结构和序列（codesign）。

核心方法: Rectified Flow Matching + 非等变 DiT + atom14 坐标表征。

详细设计见 `初步计划.md`，阶段进度见 `ROADMAP.md`。

## 环境

- Python 3.10, conda env `enzflow`, torch+cu121
- 安装: 见 `requirements.txt` 头部注释
- lint: `ruff check`, 测试: `pytest tests/ -v`

## 训练数据（不在 Git 中）

需要手动部署两样东西:
- `data/processed/enzymes/*.pt` — 蛋白质张量 (~12GB, 符号链接即可)
- `weights/ec_vectors_train.pt` — EC 条件向量 (~152MB)

## 怎么跑

```bash
# 训练 (CUDA_VISIBLE_DEVICES 控制卡数，自动检测)
python scripts/train.py
python scripts/train.py --wandb  # 开 wandb

# 参数覆盖
python scripts/train.py --lr 3e-4 --batch_size 4 --max_steps 100000
```

## 不能忘的事

- pair repr 必须用 x_t (噪声坐标) 算，**不能用 x_1**，否则信息泄漏
- 用 BF16 不用 FP16（坐标值域大，FP16 会溢出）
- BF16 张量转 numpy 要先 `.float()`
- docstring 不要用 → 等 Unicode 符号（ruff 报错）
- ruff 已 ignore N806，大写数学变量 (R, Q, H, N) 是允许的
