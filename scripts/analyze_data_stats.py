"""统计预训练蛋白质 .pt 文件的长度、体积、氨基酸种类数分布。

Usage:
    python scripts/analyze_data_stats.py [--data_dir data/processed/afdbs]
"""

import argparse
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def analyze_one(pt_path: str):
    """分析单个 .pt 文件，返回 (length, bbox_volume, bbox_xyz, n_unique_aa)。"""
    try:
        d = torch.load(pt_path, weights_only=False, map_location="cpu")
        aatype = d["aatype"]
        coords = d["coords"]        # [N, 14, 3]
        atom_mask = d["atom_mask"]   # [N, 14]

        length = len(aatype)

        # 有效原子坐标
        valid = coords[atom_mask.bool()]  # [M, 3]
        lo = valid.min(0).values.float().numpy()
        hi = valid.max(0).values.float().numpy()
        bbox = hi - lo  # [dx, dy, dz]
        volume = float(bbox[0] * bbox[1] * bbox[2])

        n_unique = int(aatype.unique().numel())

        return (length, volume, bbox.tolist(), n_unique)
    except Exception as e:
        print(f"Error processing {pt_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed/afdbs")
    parser.add_argument("--workers", type=int, default=min(16, cpu_count()))
    parser.add_argument("--output_dir", default="outputs/data_stats")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(data_dir.glob("*.pt"))
    print(f"Found {len(pt_files)} .pt files in {data_dir}")

    # 多进程分析
    pt_paths = [str(f) for f in pt_files]
    with Pool(args.workers) as pool:
        results = pool.map(analyze_one, pt_paths, chunksize=256)

    results = [r for r in results if r is not None]
    print(f"Successfully analyzed {len(results)} / {len(pt_files)} files")

    lengths = np.array([r[0] for r in results])
    volumes = np.array([r[1] for r in results])
    bboxes = np.array([r[2] for r in results])  # [N, 3]
    n_uniques = np.array([r[3] for r in results])

    # ---------- 打印统计 ----------
    print("\n=== 长度 (残基数) ===")
    print(f"  mean={lengths.mean():.1f}, median={np.median(lengths):.1f}, "
          f"std={lengths.std():.1f}")
    print(f"  min={lengths.min()}, max={lengths.max()}")
    for pct in [25, 50, 75, 90, 95, 99]:
        print(f"  P{pct}={np.percentile(lengths, pct):.0f}", end="")
    print()

    print("\n=== BBox 体积 (A^3) ===")
    print(f"  mean={volumes.mean():.0f}, median={np.median(volumes):.0f}, "
          f"std={volumes.std():.0f}")
    print(f"  min={volumes.min():.0f}, max={volumes.max():.0f}")

    # bbox 边长
    print("\n=== BBox 最大边长 (A) ===")
    max_side = bboxes.max(axis=1)
    print(f"  mean={max_side.mean():.1f}, median={np.median(max_side):.1f}, "
          f"std={max_side.std():.1f}")
    print(f"  min={max_side.min():.1f}, max={max_side.max():.1f}")

    print("\n=== 每蛋白 unique 氨基酸种类数 ===")
    print(f"  mean={n_uniques.mean():.1f}, median={np.median(n_uniques):.1f}")
    for n in range(1, 21):
        cnt = (n_uniques == n).sum()
        if cnt > 0:
            print(f"  {n} types: {cnt} proteins ({cnt/len(n_uniques)*100:.2f}%)")

    # ---------- 长度 vs bbox最大边长 散点（诊断是否被框住） ----------
    print("\n=== 长度 vs BBox 最大边长 (理论估计) ===")
    # 球状蛋白近似: 体积 ~ N * 135 A^3, 半径 ~ (3V/4pi)^(1/3)
    # 所以 bbox_side ~ 2 * (3*N*135/(4*pi))^(1/3)
    test_lengths = np.array([100, 200, 300, 400, 500])
    for L in test_lengths:
        r_est = (3 * L * 135 / (4 * np.pi)) ** (1/3)
        diameter = 2 * r_est
        actual_mask = (lengths >= L - 20) & (lengths <= L + 20)
        if actual_mask.sum() > 0:
            actual_max = np.median(max_side[actual_mask])
            print(f"  L={L}: 球状估计直径={diameter:.1f}A, "
                  f"实际bbox最大边中位数={actual_max:.1f}A (n={actual_mask.sum()})")

    # ---------- 画图 ----------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. 长度分布
    ax = axes[0, 0]
    ax.hist(lengths, bins=100, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Protein Length (residues)")
    ax.set_ylabel("Count")
    ax.set_title(f"Length Distribution (n={len(lengths)})")
    ax.axvline(np.median(lengths), color="red", ls="--",
               label=f"median={np.median(lengths):.0f}")
    ax.legend()

    # 2. BBox 体积分布
    ax = axes[0, 1]
    log_vol = np.log10(volumes + 1)
    ax.hist(log_vol, bins=100, edgecolor="black", alpha=0.7)
    ax.set_xlabel("log10(BBox Volume / A^3)")
    ax.set_ylabel("Count")
    ax.set_title("BBox Volume Distribution")
    ax.axvline(np.median(log_vol), color="red", ls="--",
               label=f"median={10**np.median(log_vol):.0f}")
    ax.legend()

    # 3. Unique AA 种类数分布
    ax = axes[0, 2]
    bins = np.arange(0.5, 21.5, 1)
    ax.hist(n_uniques, bins=bins, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Number of Unique Amino Acid Types")
    ax.set_ylabel("Count")
    ax.set_title("Unique AA Types per Protein")
    ax.set_xticks(range(1, 21))

    # 4. 长度 vs BBox 最大边长 散点
    ax = axes[1, 0]
    # 随机抽样避免太密
    idx = np.random.choice(len(lengths), min(10000, len(lengths)), replace=False)
    ax.scatter(lengths[idx], max_side[idx], s=1, alpha=0.3)
    # 理论曲线
    L_range = np.linspace(10, lengths.max(), 200)
    r_theory = (3 * L_range * 135 / (4 * np.pi)) ** (1/3)
    ax.plot(L_range, 2 * r_theory, "r-", lw=2, label="Spherical estimate")
    ax.set_xlabel("Protein Length")
    ax.set_ylabel("BBox Max Side (A)")
    ax.set_title("Length vs BBox Max Side")
    ax.legend()

    # 5. 长度 vs unique AA
    ax = axes[1, 1]
    ax.scatter(lengths[idx], n_uniques[idx], s=1, alpha=0.3)
    ax.set_xlabel("Protein Length")
    ax.set_ylabel("Unique AA Types")
    ax.set_title("Length vs Unique AA Types")

    # 6. BBox 各轴边长分布
    ax = axes[1, 2]
    for i, label in enumerate(["X", "Y", "Z"]):
        ax.hist(bboxes[:, i], bins=100, alpha=0.4, label=label)
    ax.set_xlabel("BBox Side Length (A)")
    ax.set_ylabel("Count")
    ax.set_title("BBox Side Lengths (per axis)")
    ax.legend()

    plt.tight_layout()
    fig_path = out_dir / "data_stats.png"
    plt.savefig(fig_path, dpi=150)
    print(f"\nFigure saved to {fig_path}")

    # 保存原始数据
    np.savez(out_dir / "data_stats.npz",
             lengths=lengths, volumes=volumes, bboxes=bboxes,
             n_uniques=n_uniques)
    print(f"Raw data saved to {out_dir / 'data_stats.npz'}")


if __name__ == "__main__":
    main()
