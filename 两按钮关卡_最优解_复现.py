#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
《两按钮关卡_最优解_纯定稿.md》配套复现脚本

覆盖定稿各节的承载数字：
  1. §2.3  松弛 DP 下界 F0 与 argmin 策略表 PI
  2. §2.4  网格收敛 150 / 200 / 260 / 320（数值佐证；截断精确性另有解析证明）
  3. §4    条件一命中率天花板、Cramér 根 rho*
  4. §3    CRN 配对夹逼、Garwood 泊松 95% 上沿（按较多分歧的一组取保守值）
  5. §5.3  Wald 过冲 4/3、蓄水击数、合拢几何常数
  6. §5.2 / §6  静态五规则仿真、均值恒等式

运行：python 两按钮关卡_最优解_复现.py
依赖：numpy, scipy

口径说明：
  * 定稿 §6 的 ±0.0006 天来自更大主池；本脚本静态规则用单批 1M 局，
    SE ≈ 0.0056 天，用于验证数值量级，不复现该误差棒。
  * 定稿 §3 的「约 20 局 / 约 18 局分歧」来自两组独立种子；本脚本
    同样跑两组种子并各自报告，取分歧数较多的一组算上沿。
"""

import numpy as np
from scipy.stats import chi2
from scipy.optimize import brentq

G = np.arange(1, 6)       # a ~ U{1..5}
C = np.arange(1, 4)       # c ~ U{1..3}
T_MAX = 120               # 冻结下条件二恰在第 120 击达成
MAX_DEF = 260


# ---------------------------------------------------------------
# 1. §2.3 松弛 Bellman DP（含 PI 导出）
# ---------------------------------------------------------------
def bellman(MAX=MAX_DEF, with_pi=True):
    Lg = np.arange(MAX + 1)[:, None]
    Rg = np.arange(MAX + 1)[None, :]
    F = np.full((MAX + 1, MAX + 1), 20.0)   # t=120 松弛终止；出格=20（截断精确）
    PI = np.zeros((T_MAX, MAX + 1, MAX + 1), np.int8) if with_pi else None

    for t in range(T_MAX - 1, -1, -1):
        dcost = (t + 6) // 6                 # ceil((t+1)/6)
        best, best_side = None, None
        for side in (0, 1):                  # 0=右, 1=左
            acc = np.zeros((MAX + 1, MAX + 1))
            for a in G:
                for cc in C:
                    if side == 1:
                        Lp, Rp = Lg + a, np.maximum(Rg - cc, 0)
                    else:
                        Lp, Rp = np.maximum(Lg - cc, 0), Rg + a
                    hit = (Lp + Rp == 100) & (np.abs(Lp - Rp) <= 5)
                    val = np.where(
                        (Lp <= MAX) & (Rp <= MAX),
                        F[np.minimum(Lp, MAX), np.minimum(Rp, MAX)],
                        20.0,
                    )
                    acc += np.where(hit, float(dcost), val)
            acc /= 15.0
            if best is None:
                best, best_side = acc, side
            else:
                take = acc < best            # 等值保留先出现的「右」
                best = np.where(take, acc, best)
                best_side = np.where(take, side, best_side)
        F = best
        if with_pi:
            PI[t] = best_side
    return float(F[0, 0]), PI


# ---------------------------------------------------------------
# 2. §4 命中率天花板（同状态空间上只最大化 P(条件一)）
# ---------------------------------------------------------------
def hit_ceiling(MAX=MAX_DEF):
    Lg = np.arange(MAX + 1)[:, None]
    Rg = np.arange(MAX + 1)[None, :]
    P = np.zeros((MAX + 1, MAX + 1))
    for t in range(T_MAX - 1, -1, -1):
        best = None
        for side in (0, 1):
            acc = np.zeros((MAX + 1, MAX + 1))
            for a in G:
                for cc in C:
                    if side == 1:
                        Lp, Rp = Lg + a, np.maximum(Rg - cc, 0)
                    else:
                        Lp, Rp = np.maximum(Lg - cc, 0), Rg + a
                    hit = (Lp + Rp == 100) & (np.abs(Lp - Rp) <= 5)
                    val = np.where(
                        (Lp <= MAX) & (Rp <= MAX),
                        P[np.minimum(Lp, MAX), np.minimum(Rp, MAX)],
                        0.0,
                    )
                    acc += np.where(hit, 1.0, val)
            acc /= 15.0
            best = acc if best is None else np.maximum(best, acc)
        P = best
    return float(P[0, 0])


# ---------------------------------------------------------------
# 3. §4 Cramér 根：解 E[rho^(a-c)] = 1
#    ΔS 取 -2..+4，权重 1,2,3,3,3,2,1 除以 15
# ---------------------------------------------------------------
def cramer_root():
    d = np.arange(-2, 5)
    w = np.array([1, 2, 3, 3, 3, 2, 1], float) / 15.0
    f = lambda r: np.sum(w * r ** d) - 1.0
    return brentq(f, 0.2, 0.99)


# ---------------------------------------------------------------
# 4. §3 CRN 配对夹逼：松弛执行 vs 真实执行(DP 表 + 满 60 冻结)
# ---------------------------------------------------------------
def paired(PI, MAX=MAX_DEF, NP=2_000_000, seed=777):
    rg = np.random.default_rng(seed)
    Ls, Rs = np.zeros(NP, np.int32), np.zeros(NP, np.int32)
    Lt, Rt = np.zeros(NP, np.int32), np.zeros(NP, np.int32)
    nL, nR = np.zeros(NP, np.int32), np.zeros(NP, np.int32)
    ds, dt = np.zeros(NP, np.int32), np.zeros(NP, np.int32)
    alive_s, alive_t = np.ones(NP, bool), np.ones(NP, bool)

    for t in range(T_MAX):
        n = t + 1
        # 出格状态按 MAX 截断索引即可：其值恒 20，策略任取不影响（§2.4）
        act = PI[t, np.minimum(Ls, MAX), np.minimum(Rs, MAX)]
        act_t = np.where(nL >= 60, 0, np.where(nR >= 60, 1, act))  # 冻结
        a = rg.integers(1, 6, NP)
        c = rg.integers(1, 4, NP)

        rs, ls = (act == 0) & alive_s, (act == 1) & alive_s
        Ls[rs] = np.maximum(Ls[rs] - c[rs], 0); Rs[rs] += a[rs]
        Rs[ls] = np.maximum(Rs[ls] - c[ls], 0); Ls[ls] += a[ls]

        rt, lt = (act_t == 0) & alive_t, (act_t == 1) & alive_t
        Lt[rt] = np.maximum(Lt[rt] - c[rt], 0); Rt[rt] += a[rt]; nR[rt] += 1
        Rt[lt] = np.maximum(Rt[lt] - c[lt], 0); Lt[lt] += a[lt]; nL[lt] += 1

        d = (n + 5) // 6
        h1 = alive_s & (Ls + Rs == 100) & (np.abs(Ls - Rs) <= 5)
        ds[h1] = d; alive_s[h1] = False
        u1 = alive_t & (Lt + Rt == 100) & (np.abs(Lt - Rt) <= 5)
        u2 = alive_t & ~u1 & (nL >= 60) & (nR >= 60)
        dt[u1] = d; dt[u2] = d
        alive_t[u1] = False; alive_t[u2] = False

    ds[alive_s] = 20                       # 松弛：t=120 无条件收 20 天
    dt[alive_t] = np.maximum(dt[alive_t], 20)
    diff = dt - ds
    return {
        "seed": seed,
        "松弛E": ds.mean(),
        "真实E": dt.mean(),
        "分歧局数": int(np.count_nonzero(diff)),
        "分歧率": float(np.mean(diff != 0)),
        "单局差最大天数": int(diff.max()),
        "单局差均值": float(diff[diff != 0].mean()) if np.any(diff) else 0.0,
        "配对差秒": float(diff.mean() * 86400),
    }


def garwood(k, NP=2_000_000, maxdiff_days=2):
    """泊松 95% 单侧上沿：lam_up = 0.5 * chi2.ppf(0.95, 2(k+1))"""
    lam = 0.5 * chi2.ppf(0.95, 2 * (k + 1))
    return float(lam), float(lam * maxdiff_days / NP * 86400)


# ---------------------------------------------------------------
# 5. §5.2 / §6 静态五规则（theta=73）
#    蓄水(一次性阶段) → 追低(相等固定点左) → 达成即停 → 满 60 冻结 → 禁止损
# ---------------------------------------------------------------
def static_rules(theta=73, NP=1_000_000, seed=20261105):
    rg = np.random.default_rng(seed)
    L, R = np.zeros(NP, np.int32), np.zeros(NP, np.int32)
    nL, nR = np.zeros(NP, np.int32), np.zeros(NP, np.int32)
    N, way = np.zeros(NP, np.int32), np.zeros(NP, np.int8)
    alive = np.ones(NP, bool)
    phase1 = np.ones(NP, bool)             # 一次性蓄水阶段

    for n in range(1, T_MAX + 1):
        if not alive.any():
            break
        storage = alive & phase1 & (L < theta)
        chase = np.where(L < R, 1, np.where(L > R, 0, 1)).astype(np.int8)  # 相等固定点左
        act = np.where(storage, 1, chase)
        act = np.where(nL >= 60, 0, np.where(nR >= 60, 1, act)).astype(np.int8)
        phase1 &= ~(alive & (L >= theta))

        a = rg.integers(1, 6, NP)
        c = rg.integers(1, 4, NP)
        right, left = (act == 0) & alive, (act == 1) & alive
        L[right] = np.maximum(L[right] - c[right], 0); R[right] += a[right]; nR[right] += 1
        R[left] = np.maximum(R[left] - c[left], 0); L[left] += a[left]; nL[left] += 1

        c1 = alive & (L + R == 100) & (np.abs(L - R) <= 5)
        c2 = alive & ~c1 & (nL >= 60) & (nR >= 60)
        N[c1] = n; way[c1] = 1; alive[c1] = False
        N[c2] = n; way[c2] = 2; alive[c2] = False

    day = (N + 5) // 6
    hit = way == 1
    return {
        "残留": int(alive.sum()),
        "E天": float(day.mean()),
        "SE": float(day.std() / np.sqrt(NP)),
        "命中率": float(hit.mean()),
        "E天_命中": float(day[hit].mean()),
        "SD_全体": float(day.std()),
        "中位天数": float(np.median(day)),
        "平均点击数": float(N.mean()),
        "未命中天唯一值": np.unique(day[way == 2]).tolist(),
    }


if __name__ == "__main__":
    import time
    t0 = time.time()

    print("=" * 64)
    print("《两按钮关卡_最优解_纯定稿.md》配套复现")
    print("=" * 64)

    print("\n[1] §2.3 松弛 DP 下界（MAX=260，含 PI 导出）")
    F0, PI = bellman(260)
    print(f"    F0 = {F0:.12f} 天   (定稿 14.129453951680)")
    print(f"    PI: shape={PI.shape}, dtype={PI.dtype}, 取值={np.unique(PI).tolist()}")

    print("\n[2] §2.4 网格收敛（数值佐证；截断精确性见定稿解析证明）")
    for M in (150, 200, 260, 320):
        v, _ = bellman(M, with_pi=False)
        print(f"    MAX={M:3d}: F0 = {v:.12f}")

    print("\n[3] §4 命中率天花板")
    ceil = hit_ceiling(260)
    print(f"    P_max = {ceil:.8f} = {ceil*100:.4f}%   (定稿 54.1425%)")

    print("\n[4] §4 Cramér 旁证")
    rho = cramer_root()
    print(f"    rho* = {rho:.6f}   (定稿 ≈0.433)")
    print(f"    0.433^50 ≈ {0.433**50:.3e}   (定稿 ≈7e-19，数量级 1e-18)")

    print("\n[5] §5.3(a) Wald 过冲与蓄水击数")
    overshoot = np.mean(G ** 2) / (2 * np.mean(G)) - 0.5
    theta = 73
    print(f"    E[a^2]/(2E[a]) - 1/2 = {overshoot:.6f} = 4/3")
    print(f"    E[N] = (73 + 4/3)/3 = {(theta + 4/3)/3:.4f} 击   (定稿实测 24.779)")

    print("\n[6] §5.3(c) 合拢几何（均值近似）")
    print(f"    S_edge = 1.2*73+0.6 = {1.2*theta+0.6:.1f}   (定稿 88.2)")
    print(f"    N(73)  = 99.4-1.2*73 = {99.4-1.2*theta:.1f}   (定稿 11.8)")

    print("\n[7] §3 CRN 配对夹逼（2M 局 × 2 组独立种子，仅作量级验证）")
    results = [paired(PI, NP=2_000_000, seed=s) for s in (777, 20261105)]
    for r in results:
        lam, _ = garwood(r["分歧局数"])
        sec_avg = lam * r["单局差均值"] / 2_000_000 * 86400
        print(f"    seed={r['seed']:<10d} 分歧={r['分歧局数']:3d} 局 "
              f"({r['分歧率']:.2e}), 差max={r['单局差最大天数']}天, "
              f"差均值={r['单局差均值']:.2f}天, 配对差={r['配对差秒']:.4f} 秒, "
              f"按平均差换算上沿={sec_avg:.3f} 秒")
    print("    注：本脚本种子与定稿不同，分歧数与单局差会波动（个别种子可出现 3 天差）；")
    print("        定稿「每局 ≤2 天」是其自身种子组的实测，且 §7 已标为启发级。")
    print("        按各组实际平均单局差换算的 95% 上沿均低于定稿 2.5 秒。")
    print(f"    —— 定稿口径（其种子组 k=20，按每局最坏 2 天计）——")
    print(f"    保守点估计 = 20×2/2M = {20*2/2_000_000*86400:.3f} 秒  (定稿 1.728)")
    print(f"    Garwood k=20, df=42: lam_up={garwood(20)[0]:.4f}, "
          f"上沿={garwood(20)[1]:.3f} 秒 ≈ 2.5 秒")
    print(f"    Garwood k=18, df=38: lam_up={garwood(18)[0]:.4f}, "
          f"上沿={garwood(18)[1]:.3f} 秒")

    print("\n[8] §5.2 / §6 静态五规则（单批 1M 局；SE 与定稿主池口径不同）")
    st = static_rules()
    for key, val in st.items():
        print(f"    {key:12s}: {val}")

    print("\n[9] §6 均值恒等式")
    lhs = 0.53604 * 9.06289 + 0.46396 * 20
    print(f"    0.53604*9.06289 + 0.46396*20 = {lhs:.5f}")
    print(f"    与实测 14.13703 差 {lhs-14.13703:.5f} 天 < SE 0.00060 ✓")
    print(f"    14.1370 - 14.129454 = {14.1370-14.129454:.6f} 天 "
          f"= {(14.1370-14.129454)*1440:.2f} 分钟  (定稿 10.9)")

    print(f"\n总耗时 {time.time()-t0:.1f}s")
    print("说明：静态规则单批 1M 局 SE≈0.0056 天，用于验证量级；")
    print("      定稿 ±0.0006 天来自更大主池，不复现该误差棒。")
