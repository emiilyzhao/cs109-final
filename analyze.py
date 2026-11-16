"""Fit a tap-rhythm authentication model and evaluate it with an ROC curve.

Model
-----
An attempt with k taps gives d = k - 1 gaps. We work with log-gaps
x = log(gaps), because gaps are positive and right-skewed, and timing
noise is roughly proportional to the gap length.

    owner model:     x ~ N(mu_o, Sigma_o)   fitted by MLE on genuine attempts
    impostor model:  x ~ N(mu_i, Sigma_i)   fitted by MLE on impostor attempts

Decision rule (likelihood ratio test):
    score(x) = log p(x | owner) - log p(x | impostor)
    accept if score(x) > threshold

Sweeping the threshold traces out the ROC curve: true accept rate
(genuine attempts accepted) against false accept rate (impostors accepted).

All scores are computed leave-one-out: an attempt is never scored by a
model that was fitted on it, so the ROC reflects performance on new taps.

Usage:
    python analyze.py taps.csv                 # your recorded data
    python analyze.py --simulate               # synthetic data, no hardware needed
"""
import argparse
import csv
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# ---------------------------------------------------------------- data

def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            gaps = [float(g) for g in r["gaps_ms"].split(";") if g]
            rows.append((r["tapper"], r["rhythm"], np.array(gaps)))
    return rows


def simulate(path, seed=0):
    """Write synthetic attempts: an owner plus three copycats."""
    rng = np.random.default_rng(seed)
    base = np.array([180, 180, 360, 180, 540, 180, 180.0])  # the owner's rhythm, in ms
    rows = []
    # Genuine: a shared tempo factor per attempt (this makes the gaps
    # correlated) times small independent per-gap jitter.
    for _ in range(40):
        tempo = rng.lognormal(0, 0.06)
        rows.append(("owner", "owner", base * tempo * rng.lognormal(0, 0.05, base.size)))
    # Copycats: each has a persistent distortion of the rhythm, plus more jitter.
    for name in ["alex", "sam", "jordan"]:
        distortion = rng.lognormal(0, 0.12, base.size)
        for _ in range(12):
            tempo = rng.lognormal(0, 0.10)
            rows.append((name, "owner", base * distortion * tempo * rng.lognormal(0, 0.08, base.size)))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tapper", "rhythm", "gaps_ms"])
        for t, r, g in rows:
            w.writerow([t, r, ";".join(f"{x:.1f}" for x in g)])
    print(f"Wrote {len(rows)} simulated attempts to {path}")


# ---------------------------------------------------------------- model

def fit_mvn(X, shrink=0.2):
    """MLE mean and covariance, with the covariance shrunk toward its diagonal.

    With few attempts the raw MLE covariance is noisy or singular.
    Shrinkage acts like a prior that gaps are uncorrelated, and is
    worth discussing in the report as a MAP-style regularizer.
    """
    mu = X.mean(axis=0)
    S = np.cov(X, rowvar=False, bias=True)  # bias=True gives the MLE (divide by n)
    S = (1 - shrink) * S + shrink * np.diag(np.diag(S))
    S += 1e-6 * np.eye(len(mu))
    return mu, S


def logpdf(x, mu, S):
    return stats.multivariate_normal.logpdf(x, mean=mu, cov=S)


def loo_logpdf(X, i, shrink):
    """Log density of row i under a model fitted on every other row."""
    mu, S = fit_mvn(np.delete(X, i, axis=0), shrink)
    return logpdf(X[i], mu, S)


def roc(genuine, impostor):
    """True and false accept rates for every threshold, plus AUC and EER."""
    thresholds = np.sort(np.concatenate([genuine, impostor]))[::-1]
    tar = np.array([(genuine >= t).mean() for t in thresholds])
    far = np.array([(impostor >= t).mean() for t in thresholds])
    tar = np.concatenate([[0], tar, [1]])
    far = np.concatenate([[0], far, [1]])
    auc = np.trapezoid(tar, far)
    frr = 1 - tar
    k = np.argmin(np.abs(frr - far))
    eer = (frr[k] + far[k]) / 2
    return far, tar, auc, eer


# ---------------------------------------------------------------- analysis

def analyze(rows, owner, shrink, use_log, out_png):
    rows = [r for r in rows if r[1] == owner]
    # Keep only attempts with the usual number of taps (mis-taps are rejected outright).
    d = Counter(len(g) for _, _, g in rows).most_common(1)[0][0]
    dropped = sum(len(g) != d for _, _, g in rows)
    rows = [r for r in rows if len(r[2]) == d]

    feat = (lambda g: np.log(g)) if use_log else (lambda g: g)
    G = np.array([feat(g) for t, _, g in rows if t == owner])
    I = np.array([feat(g) for t, _, g in rows if t != owner])
    print(f"Rhythm '{owner}': {d + 1} taps per attempt, {len(G)} genuine, "
          f"{len(I)} impostor attempts ({dropped} dropped for a wrong tap count)")
    if len(G) < 5 or len(I) < 3:
        raise SystemExit("Need at least 5 genuine and 3 impostor attempts.")

    # Is each gap plausibly Gaussian? (Shapiro-Wilk on genuine attempts)
    raw = np.exp(G) if use_log else G
    p_raw = [stats.shapiro(raw[:, j]).pvalue for j in range(d)]
    p_log = [stats.shapiro(np.log(raw[:, j])).pvalue for j in range(d)]
    print(f"Gaps consistent with Gaussian (p > 0.05): raw {sum(p > 0.05 for p in p_raw)}/{d}, "
          f"log {sum(p > 0.05 for p in p_log)}/{d}")

    # Leave-one-out scores
    mu_o, S_o = fit_mvn(G, shrink)
    mu_i, S_i = fit_mvn(I, shrink)
    llr_gen = np.array([loo_logpdf(G, k, shrink) - logpdf(G[k], mu_i, S_i) for k in range(len(G))])
    llr_imp = np.array([logpdf(I[k], mu_o, S_o) - loo_logpdf(I, k, shrink) for k in range(len(I))])
    # Baseline: owner likelihood alone, with no impostor model
    own_gen = np.array([loo_logpdf(G, k, shrink) for k in range(len(G))])
    own_imp = np.array([logpdf(x, mu_o, S_o) for x in I])

    far, tar, auc, eer = roc(llr_gen, llr_imp)
    far_b, tar_b, auc_b, eer_b = roc(own_gen, own_imp)
    print(f"Likelihood ratio test: AUC = {auc:.3f}, equal error rate = {eer:.1%}")
    print(f"Owner likelihood only: AUC = {auc_b:.3f}, equal error rate = {eer_b:.1%}")

    # Figures
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))
    gaps_g, gaps_i = (np.exp(G), np.exp(I)) if use_log else (G, I)
    idx = np.arange(1, d + 1)
    ax[0, 0].errorbar(idx - 0.1, gaps_g.mean(0), gaps_g.std(0), fmt="o-", capsize=3, label="owner")
    ax[0, 0].errorbar(idx + 0.1, gaps_i.mean(0), gaps_i.std(0), fmt="s--", capsize=3, label="impostors")
    ax[0, 0].set(title="Rhythm profile (mean ± 1 sd)", xlabel="gap number", ylabel="gap (ms)")
    ax[0, 0].legend()

    corr = np.corrcoef(G, rowvar=False)
    im = ax[0, 1].imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax[0, 1].set(title="Correlation between owner's gaps", xticks=range(d), yticks=range(d),
                 xticklabels=idx, yticklabels=idx)
    fig.colorbar(im, ax=ax[0, 1])

    bins = np.linspace(min(llr_gen.min(), llr_imp.min()), max(llr_gen.max(), llr_imp.max()), 30)
    ax[1, 0].hist(llr_gen, bins, alpha=0.6, label="genuine")
    ax[1, 0].hist(llr_imp, bins, alpha=0.6, label="impostor")
    ax[1, 0].set(title="Log-likelihood ratio scores", xlabel="log p(x|owner) − log p(x|impostor)",
                 ylabel="attempts")
    ax[1, 0].legend()

    ax[1, 1].plot(far, tar, label=f"likelihood ratio (AUC {auc:.3f}, EER {eer:.1%})")
    ax[1, 1].plot(far_b, tar_b, "--", label=f"owner likelihood only (AUC {auc_b:.3f})")
    ax[1, 1].plot([0, 1], [0, 1], ":", color="gray")
    ax[1, 1].set(title="ROC curve", xlabel="false accept rate", ylabel="true accept rate")
    ax[1, 1].legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"Saved figure to {out_png}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", nargs="?", default="taps.csv")
    p.add_argument("--simulate", action="store_true", help="generate synthetic data first")
    p.add_argument("--owner", help="whose rhythm to evaluate (default: most common)")
    p.add_argument("--shrink", type=float, default=0.2, help="covariance shrinkage, 0 to 1")
    p.add_argument("--raw", action="store_true", help="use raw gaps instead of log-gaps")
    p.add_argument("--out", default="tap_lock_results.png")
    args = p.parse_args()

    if args.simulate:
        args.csv = "taps_sim.csv"
        simulate(args.csv)
    rows = load(args.csv)
    owner = args.owner or Counter(r for _, r, _ in rows).most_common(1)[0][0]
    analyze(rows, owner, args.shrink, not args.raw, args.out)


if __name__ == "__main__":
    main()
