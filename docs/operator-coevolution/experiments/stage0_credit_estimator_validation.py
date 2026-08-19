"""
Stage 0: 零成本合成验证 —— 算子对信用分配的估计器 / 采样策略对比

背景
----
G-LNS (arXiv:2602.08253) 与 CoEvo-AHD (arXiv:2606.00718) 在协同进化 destroy-repair
算子时, 都用同一个标量奖励同时更新三个账户:

    F(d_i) += sigma      # 破坏算子适应度 (用于剪枝)
    F(r_j) += sigma      # 修复算子适应度
    S_ij   += sigma      # "协同矩阵" (用于 joint crossover 选对)

本脚本在**已知 ground truth** 的合成 payoff 上检验三件事, 全程零 LLM 调用、零 LNS 运行。

    E1 estimators : S_ij 到底测的是交互还是主效应? 方差分解能不能修好?
    E2 coverage   : 自适应选择 (roulette/Thompson) 会不会饿死交互项的估计?
    E3 coldstart  : 累加式 F 在种群更替时对新算子是否系统性不公?

真实模型
--------
    M[i,j] = mu + alpha_i + beta_j + gamma_ij
    alpha/beta = 主效应 (算子自身好坏)
    gamma      = 交互项 = 真正意义上的"配对协同"

用法
----
    python docs/operator-coevolution/experiments/stage0_credit_estimator_validation.py --experiment estimators
    python docs/operator-coevolution/experiments/stage0_credit_estimator_validation.py --experiment coverage
    python docs/operator-coevolution/experiments/stage0_credit_estimator_validation.py --experiment coldstart
    python docs/operator-coevolution/experiments/stage0_credit_estimator_validation.py --experiment all
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import spearmanr


# ============================================================================
# 1. Ground truth
# ============================================================================

@dataclass
class PayoffGroundTruth:
    n_destroy: int
    n_repair: int
    mu: float
    alpha: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray
    noise_std: float

    @property
    def M(self) -> np.ndarray:
        return self.mu + self.alpha[:, None] + self.beta[None, :] + self.gamma

    def sample(self, i: int, j: int, rng) -> float:
        return float(self.M[i, j] + rng.normal(0.0, self.noise_std))

    def gamma_var_share(self) -> float:
        tot = np.var(self.M)
        return float(np.var(self.gamma) / tot) if tot > 1e-12 else 0.0


def make_ground_truth(nd, nr, gamma_scale, noise_std, rng) -> PayoffGroundTruth:
    """gamma_scale 控制交互强度: 0=完全可加(无协同可言), 1=交互与主效应同量级"""
    alpha = rng.normal(0, 1.0, nd); alpha -= alpha.mean()
    beta = rng.normal(0, 1.0, nr); beta -= beta.mean()
    gamma = rng.normal(0, gamma_scale, (nd, nr))
    gamma -= gamma.mean(axis=0, keepdims=True)   # 双向中心化,
    gamma -= gamma.mean(axis=1, keepdims=True)   # 把混进 gamma 的主效应挤出去
    return PayoffGroundTruth(nd, nr, 0.0, alpha, beta, gamma, noise_std)


# ============================================================================
# 2. 采样策略
# ============================================================================

@dataclass
class SampleLog:
    i: list = field(default_factory=list)
    j: list = field(default_factory=list)
    r: list = field(default_factory=list)
    p_sel: list = field(default_factory=list)

    def arrays(self):
        return (np.asarray(self.i), np.asarray(self.j),
                np.asarray(self.r), np.asarray(self.p_sel))


def _softmax(x, floor):
    e = np.exp(x - x.max()); p = e / e.sum()
    p = np.maximum(p, floor); return p / p.sum()


def run_sampling(gt, n_steps, selection, rng, explore_eps=0.0,
                 lam=0.9, p_floor=1e-3) -> SampleLog:
    """explore_eps: 以该概率转为"补最稀疏格子"的定向探索 (targeted coverage)。

    这一项是本脚本的核心操作变量 —— 它把"配对选择"从纯利用变成
    "利用 + 面向交互项估计的主动实验设计"。
    """
    nd, nr = gt.n_destroy, gt.n_repair
    log = SampleLog()
    cell_cnt = np.zeros((nd, nr))

    w_d, w_r = np.ones(nd), np.ones(nr)
    s_d, c_d = np.zeros(nd), np.zeros(nd)
    s_r, c_r = np.zeros(nr), np.zeros(nr)

    for _ in range(n_steps):
        if selection == "roulette":
            pd_ = np.maximum(w_d, p_floor); pd_ /= pd_.sum()
            pr_ = np.maximum(w_r, p_floor); pr_ /= pr_.sum()
        elif selection == "thompson":
            mu_d = np.where(c_d > 0, s_d / np.maximum(c_d, 1), 0.0)
            mu_r = np.where(c_r > 0, s_r / np.maximum(c_r, 1), 0.0)
            pd_ = _softmax(3.0 * (mu_d + rng.normal(0, 1 / np.sqrt(c_d + 1))), p_floor)
            pr_ = _softmax(3.0 * (mu_r + rng.normal(0, 1 / np.sqrt(c_r + 1))), p_floor)
        elif selection == "uniform":
            pd_ = np.full(nd, 1.0 / nd); pr_ = np.full(nr, 1.0 / nr)
        else:
            raise ValueError(selection)

        p_joint = np.outer(pd_, pr_)
        if explore_eps > 0:
            # 定向探索: 概率反比于该格已有样本数, 优先补最欠采样的配对
            inv = 1.0 / (1.0 + cell_cnt)
            p_joint = (1 - explore_eps) * p_joint + explore_eps * inv / inv.sum()

        flat = rng.choice(nd * nr, p=p_joint.ravel())
        i, j = divmod(flat, nr)
        r = gt.sample(i, j, rng)

        log.i.append(i); log.j.append(j); log.r.append(r)
        log.p_sel.append(float(p_joint[i, j]))
        cell_cnt[i, j] += 1

        if selection == "roulette":
            w_d[i] = lam * w_d[i] + (1 - lam) * max(r, 0.0)
            w_r[j] = lam * w_r[j] + (1 - lam) * max(r, 0.0)
        elif selection == "thompson":
            s_d[i] += r; c_d[i] += 1
            s_r[j] += r; c_r[j] += 1

    return log


# ============================================================================
# 3. 估计器
# ============================================================================

def est_naive_sum(log, nd, nr):
    """G-LNS 原方案: 纯累加"""
    i, j, r, _ = log.arrays()
    F_d = np.zeros(nd); F_r = np.zeros(nr); S = np.zeros((nd, nr))
    np.add.at(F_d, i, r); np.add.at(F_r, j, r); np.add.at(S, (i, j), r)
    return F_d, F_r, S


def est_naive_mean(log, nd, nr):
    i, j, r, _ = log.arrays()
    F_d = np.zeros(nd); c1 = np.zeros(nd)
    F_r = np.zeros(nr); c2 = np.zeros(nr)
    S = np.zeros((nd, nr)); c3 = np.zeros((nd, nr))
    np.add.at(F_d, i, r); np.add.at(c1, i, 1)
    np.add.at(F_r, j, r); np.add.at(c2, j, 1)
    np.add.at(S, (i, j), r); np.add.at(c3, (i, j), 1)
    return F_d / np.maximum(c1, 1), F_r / np.maximum(c2, 1), S / np.maximum(c3, 1)


def _design(i, j, nd, nr):
    """sum-to-zero 编码: [1 | alpha | beta | gamma]"""
    n = len(i)
    Xa = np.zeros((n, nd - 1)); Xb = np.zeros((n, nr - 1))
    for k in range(nd - 1):
        Xa[:, k] = (i == k).astype(float) - (i == nd - 1).astype(float)
    for k in range(nr - 1):
        Xb[:, k] = (j == k).astype(float) - (j == nr - 1).astype(float)
    Xg = np.einsum("nk,nl->nkl", Xa, Xb).reshape(n, -1)
    return np.hstack([np.ones((n, 1)), Xa, Xb, Xg])


def _unpack(coef, nd, nr):
    mu = coef[0]
    a = np.zeros(nd); a[:nd - 1] = coef[1:nd]; a[nd - 1] = -a[:nd - 1].sum()
    off = nd
    b = np.zeros(nr); b[:nr - 1] = coef[off:off + nr - 1]; b[nr - 1] = -b[:nr - 1].sum()
    off += nr - 1
    g_red = coef[off:].reshape(nd - 1, nr - 1)
    g = np.zeros((nd, nr))
    g[:nd - 1, :nr - 1] = g_red
    g[nd - 1, :nr - 1] = -g_red.sum(axis=0)
    g[:nd - 1, nr - 1] = -g_red.sum(axis=1)
    g[nd - 1, nr - 1] = g_red.sum()
    return mu, a, b, g


def est_anova(log, nd, nr, ridge=1.0, use_ips=False):
    """Delta_ij = mu + alpha_i + beta_j + gamma_ij, 岭回归求解。

    use_ips: 用 1/p_sel 加权修正自适应采样的选择偏差。
    """
    i, j, r, p = log.arrays()
    X = _design(i, j, nd, nr)
    if use_ips:
        w = 1.0 / np.maximum(p, 1e-6)
        w = w / w.mean()
        w = np.minimum(w, np.quantile(w, 0.99))
    else:
        w = np.ones(len(r))
    sw = np.sqrt(w)[:, None]
    Xw, yw = X * sw, r * np.sqrt(w)
    A = Xw.T @ Xw + ridge * np.eye(X.shape[1]); A[0, 0] -= ridge
    return _unpack(np.linalg.solve(A, Xw.T @ yw), nd, nr)


def _sp(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(spearmanr(a, b).statistic)


# ============================================================================
# E1: 估计器对比
# ============================================================================

def exp_estimators(cfg):
    print(f"\n{'='*78}")
    print("  E1  S_ij 测的是交互还是主效应?  方差分解能不能修好?")
    print(f"  {cfg.n_destroy}x{cfg.n_repair} 算子 | {cfg.n_steps} 次评估 | "
          f"gamma_scale={cfg.gamma_scale} | 重复 {cfg.n_repeat}")
    print(f"{'='*78}")

    for selection in ["uniform", "roulette", "thompson"]:
        rows = []
        for seed in range(cfg.n_repeat):
            rng = np.random.default_rng(seed)
            gt = make_ground_truth(cfg.n_destroy, cfg.n_repair,
                                   cfg.gamma_scale, cfg.noise_std, rng)
            log = run_sampling(gt, cfg.n_steps, selection, rng)
            main = gt.alpha[:, None] + gt.beta[None, :]
            _, _, S_sum = est_naive_sum(log, gt.n_destroy, gt.n_repair)
            _, a_ols, _, g_ols = est_anova(log, gt.n_destroy, gt.n_repair, use_ips=False)
            _, a_ips, _, g_ips = est_anova(log, gt.n_destroy, gt.n_repair, use_ips=True)
            best = np.unravel_index(np.argmax(gt.gamma), gt.gamma.shape)
            n_cov = int((np.bincount(log.arrays()[0] * gt.n_repair + log.arrays()[1],
                                     minlength=gt.n_destroy * gt.n_repair) > 0).sum())
            rows.append(dict(
                S_vs_gamma=_sp(S_sum, gt.gamma),
                S_vs_main=_sp(S_sum, main),
                anova_gamma=_sp(g_ols, gt.gamma),
                anova_gamma_ips=_sp(g_ips, gt.gamma),
                anova_alpha=_sp(a_ols, gt.alpha),
                pick_S=float(np.unravel_index(np.argmax(S_sum), S_sum.shape) == best),
                pick_anova=float(np.unravel_index(np.argmax(g_ols), g_ols.shape) == best),
                coverage=n_cov / (gt.n_destroy * gt.n_repair),
            ))
        agg = {k: np.mean([r[k] for r in rows]) for k in rows[0]}
        print(f"\n  --- 采样策略: {selection}  (payoff 矩阵覆盖率 {agg['coverage']:.0%}) ---")
        print(f"    S_ij  vs 真实 gamma   (声称测的)      {agg['S_vs_gamma']:+.3f}")
        print(f"    S_ij  vs alpha+beta   (实际测的)      {agg['S_vs_main']:+.3f}")
        print(f"    ANOVA gamma_hat vs 真实 gamma         {agg['anova_gamma']:+.3f}")
        print(f"    ANOVA gamma_hat vs 真值 (+IPS)        {agg['anova_gamma_ips']:+.3f}")
        print(f"    ANOVA alpha_hat vs 真实 alpha         {agg['anova_alpha']:+.3f}")
        print(f"    最优协同对命中率:  argmax S = {agg['pick_S']:.0%}"
              f"   |   argmax gamma_hat = {agg['pick_anova']:.0%}")


# ============================================================================
# E2: 利用 / 覆盖 的权衡  <- 核心发现
# ============================================================================

def exp_coverage(cfg):
    print(f"\n{'='*78}")
    print("  E2  自适应选择会饿死交互项的估计: 利用 vs. 覆盖 的权衡")
    print("      explore_eps = 转为'补最稀疏配对'的概率 (面向 gamma 估计的主动实验设计)")
    print(f"{'='*78}")
    # 用**紧预算**跑, 否则格子数少时随便采都能采满, 看不出问题。
    # 真实场景: 每代只有有限个 LNS episode, 而算子池每代都在换。
    n_steps = max(cfg.n_steps // 5, cfg.n_destroy * cfg.n_repair * 2)
    print(f"\n  采样策略={cfg.selection} | 预算={n_steps} 次评估 "
          f"({cfg.n_destroy}x{cfg.n_repair}={cfg.n_destroy*cfg.n_repair} 个配对格)")
    print(f"\n  {'eps':>6} | {'平均奖励':>9} | {'矩阵覆盖':>8} | {'gamma 恢复':>10} | {'最优对命中':>10}")
    print(f"  {'-'*6}-+-{'-'*9}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}")

    for eps in [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]:
        rew, cov, gam, pick = [], [], [], []
        for seed in range(cfg.n_repeat):
            rng = np.random.default_rng(seed)
            gt = make_ground_truth(cfg.n_destroy, cfg.n_repair,
                                   cfg.gamma_scale, cfg.noise_std, rng)
            log = run_sampling(gt, n_steps, cfg.selection, rng, explore_eps=eps)
            i, j, r, _ = log.arrays()
            _, _, _, g = est_anova(log, gt.n_destroy, gt.n_repair)
            best = np.unravel_index(np.argmax(gt.gamma), gt.gamma.shape)
            ncell = gt.n_destroy * gt.n_repair
            rew.append(r.mean())
            cov.append((np.bincount(i * gt.n_repair + j, minlength=ncell) > 0).sum() / ncell)
            gam.append(_sp(g, gt.gamma))
            pick.append(float(np.unravel_index(np.argmax(g), g.shape) == best))
        print(f"  {eps:>6.2f} | {np.mean(rew):>+9.3f} | {np.mean(cov):>7.0%} | "
              f"{np.mean(gam):>+10.3f} | {np.mean(pick):>9.0%}")
    print("\n  解读: eps=0 (纯利用, 即现有做法) 的累计奖励最高, 但 gamma 估计最差。")
    print("        协同进化的目的是**产出好算子对**而不是最大化评估期奖励,")
    print("        所以为 gamma 估计牺牲一部分即时奖励是划算的 —— 这正是现有方法漏掉的。")


# ============================================================================
# E3: 种群更替时的冷启动不公
# ============================================================================

def exp_coldstart(cfg):
    """新算子刚进种群时样本极少, 任何纯经验估计都会把它误杀。

    这不是**偏差**问题 (换成均值也救不了), 而是**方差/样本量**问题。
    唯一的解法是把估计朝一个信息性先验收缩:
        alpha_hat = (n * empirical_mean + kappa * prior) / (n + kappa)
    这个 prior 正是"探针语义指纹 / LLM 语义先验"该发挥作用的地方。

    本实验回答一个很实际的问题: 先验要多准才值得做?
    """
    print(f"\n{'='*78}")
    print("  E3  新算子冷启动: 剪枝误杀 与 先验收缩的价值")
    print("      场景: 跑 n_steps 后把 3 个最差算子替换成真实质量在 85 分位的新算子,")
    print("            再跑 n_steps//4 (新算子只拿到很少样本), 然后按分数剪掉最差 3 个")
    print(f"{'='*78}")

    n_new, kappa = 3, 8.0
    prior_qualities = [0.0, 0.3, 0.5, 0.7, 0.9]
    res = {"F_sum": [], "F_mean": [], "anova": []}
    res_prior = {q: [] for q in prior_qualities}

    for seed in range(cfg.n_repeat):
        rng = np.random.default_rng(seed)
        gt = make_ground_truth(cfg.n_destroy, cfg.n_repair,
                               cfg.gamma_scale, cfg.noise_std, rng)
        log = run_sampling(gt, cfg.n_steps, cfg.selection, rng)

        worst = np.argsort(gt.alpha)[:n_new]
        gt.alpha[worst] = np.quantile(gt.alpha, 0.85)
        gt.gamma[worst] = rng.normal(0, cfg.gamma_scale, (n_new, gt.n_repair))

        # 第二阶段预算很小 (每个破坏算子平均只有 ~8 次机会), 老算子保留历史,
        # 新算子从零开始 —— 这正是 G-LNS 不做 reset 时的情形。
        log2 = run_sampling(gt, 8 * gt.n_destroy, cfg.selection, rng)
        keep = [k for k, ii in enumerate(log.i) if ii not in set(worst.tolist())]
        merged = SampleLog(
            i=[log.i[k] for k in keep] + list(log2.i),
            j=[log.j[k] for k in keep] + list(log2.j),
            r=[log.r[k] for k in keep] + list(log2.r),
            p_sel=[log.p_sel[k] for k in keep] + list(log2.p_sel),
        )
        log = merged

        i, _, r, _ = log.arrays()
        cnt = np.bincount(i, minlength=gt.n_destroy).astype(float)
        F_sum, _, _ = est_naive_sum(log, gt.n_destroy, gt.n_repair)
        F_mean, _, _ = est_naive_mean(log, gt.n_destroy, gt.n_repair)
        _, a_hat, _, _ = est_anova(log, gt.n_destroy, gt.n_repair)

        def kill_rate(score):
            doomed = set(np.argsort(score)[:n_new].tolist())
            return len(doomed & set(worst.tolist())) / n_new

        res["F_sum"].append(kill_rate(F_sum))
        res["F_mean"].append(kill_rate(F_mean))
        res["anova"].append(kill_rate(a_hat))

        # 先验收缩: prior 是真实 alpha 的带噪版本, prior_q 控制其与真值的相关性
        for q in prior_qualities:
            if q <= 0:
                prior = np.zeros(gt.n_destroy)
            else:
                nz = np.sqrt(max(1.0 / q**2 - 1.0, 0.0))
                prior = gt.alpha + rng.normal(0, nz * gt.alpha.std(), gt.n_destroy)
            shrunk = (cnt * a_hat + kappa * prior) / (cnt + kappa)
            res_prior[q].append(kill_rate(shrunk))

    print(f"\n  新算子真实质量位于 85 分位, 却被剪枝误杀的比例 (越低越好):")
    print(f"\n  [A] 估计器对比")
    print(f"    F 累加   (G-LNS 原方案)                {np.mean(res['F_sum']):.0%}"
          f"   <- 新算子几乎必被误杀")
    print(f"    F 均值   (频率去偏)                    {np.mean(res['F_mean']):.0%}")
    print(f"    ANOVA alpha_hat                        {np.mean(res['anova']):.0%}"
          f"   <- 跨伙伴维度借统计强度")
    print(f"\n  [B] 在 ANOVA 之上再向先验收缩 (kappa={kappa:.0f})")
    print(f"    {'先验与真实质量的相关性':<26s} {'误杀率':>8s}")
    for q in prior_qualities:
        tag = "  (无先验/收缩向 0)" if q == 0 else ""
        print(f"    rho ~ {q:.1f}{tag:<21s} {np.mean(res_prior[q]):>7.0%}")
    print("\n  解读:")
    print("    1) 累加式 F 与'被选中次数'几乎同义, 新算子样本少 => 排名垫底 => 被误杀。")
    print("       G-LNS 用'每代清零 F 和 S'来打补丁, 代价是丢掉全部跨代信息。")
    print("    2) 改成均值只解决一半; **方差分解**才是真正的解 —— 因为它能跨伙伴维度")
    print("       借统计强度: 新破坏算子的少量样本会按'碰到的是哪些修复算子'做校正。")
    print("       这是支持 alpha+beta+gamma 分解的第三个独立论据 (前两个见 E1/E2)。")
    print("    3) 再叠一个信息性先验还能继续降低误杀 —— 这给'探针语义指纹 / LLM 语义先验'")
    print("       一个可量化的价值靶子: 先验需要多准才值得你去实现。")


# ============================================================================

@dataclass
class Config:
    n_destroy: int = 8
    n_repair: int = 8
    n_steps: int = 3000
    gamma_scale: float = 0.7
    noise_std: float = 1.0
    selection: str = "roulette"
    n_repeat: int = 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=["estimators", "coverage", "coldstart", "all"],
                    default="all")
    ap.add_argument("--n-destroy", type=int, default=8)
    ap.add_argument("--n-repair", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=3000)
    ap.add_argument("--gamma-scale", type=float, default=0.7)
    ap.add_argument("--noise-std", type=float, default=1.0)
    ap.add_argument("--selection", choices=["roulette", "thompson", "uniform"],
                    default="roulette")
    ap.add_argument("--n-repeat", type=int, default=30)
    args = vars(ap.parse_args())
    which = args.pop("experiment")
    cfg = Config(**args)

    if which in ("estimators", "all"):
        exp_estimators(cfg)
    if which in ("coverage", "all"):
        exp_coverage(cfg)
    if which in ("coldstart", "all"):
        exp_coldstart(cfg)
    print()


if __name__ == "__main__":
    main()
