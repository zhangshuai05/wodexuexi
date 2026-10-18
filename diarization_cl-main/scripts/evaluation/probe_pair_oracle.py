# -*- coding: utf-8 -*-
"""撞人对的可分性 oracle：CAM++ 里到底有没有"分开这对人"的信息。

上一轮的分组把病灶钉在聚类上，不在边界上：A 腿（参考标签）内部帧错误 0.9%，
B 腿（AHC@真值K）内部 17.7%；A-B 内部差与该场"最像一对"的质心余弦相关 r=0.755，
阈值约 0.5——余弦 <0.5 的 6 场损失 0.0~6.7 点，>=0.5 的 6 场损失 19.4~47.8 点。

剩下唯一要回答的问题：这对人分不开，是因为

  (i)  信息不在表示里（192 维 CAM++ 把两人叠在一起）-> 换嵌入模型
  (ii) 信息在，但**裸余弦不是正确的度量**          -> 加打分后端，零新推理

判据：用**参考标签**在这批纯窗上做有监督线性变换，跨场拟合、目标场回评，
看该对的 EER 掉不掉。掉到很低 -> 信息在，走 (ii)；掉不下来 -> 信息不在，走 (i)。

三种变换（同一批 pair）：
  raw      裸余弦（现状基线）
  whiten   全协方差白化（无监督，只用训练集的窗）
  lda      线性判别（有监督，类=说话人）
  pca+lda  LDA 前置 PCA 降维（d=192 而样本常 < d，直接 LDA 会退化）

三种训练集（同一批 pair）：
  within   目标场自己劈一半（乐观上限：同一录音里学，只回答"信息在不在"）
  group    同一身份组其它会议的**同一对**说话人（跨场同人，oracle 式）
  global   其它会议的**全部**说话人（这才是能部署的后端：不挑人）

`within` / `group` 都是 oracle 式（明知是哪一对），只用来判信息在不在；
`global` 不挑人，是真正对应"加后端"那个方案的数。holdout 上的验证留到之后。

零推理，全部走 `w2_h1.npz` 缓存。

    python scripts/evaluation/probe_pair_oracle.py --split dev --seconds 600
"""
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "streaming_asr"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from v12_aishell4 import meeting_group, split_meetings  # noqa: E402

PURE_VOTES = 1.0


def load(cache, meeting, window, hop):
    path = Path(cache) / meeting / f"w{window:g}_h{hop:g}.npz"
    data = np.load(path, allow_pickle=False)
    return (data["emb"], np.asarray(data["times"], dtype=float),
            data["speakers"].astype(str), np.asarray(data["votes"], dtype=float))


def pure_windows(emb, times, speakers, votes):
    """只留窗内只有一位参考说话人的窗——标签可信，撞人对的检验才干净。"""
    keep = (votes >= PURE_VOTES) & (speakers != "")
    return emb[keep], times[keep], speakers[keep]


def l2norm(X):
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-9)


def eer(same, diff):
    """等错误率。分数越高越像同一人。"""
    same, diff = np.sort(np.asarray(same)), np.sort(np.asarray(diff))
    if len(same) < 2 or len(diff) < 2:
        return None
    thresholds = np.unique(np.concatenate([same, diff]))
    fnr = np.searchsorted(same, thresholds, side="left") / len(same)
    fmr = 1.0 - np.searchsorted(diff, thresholds, side="right") / len(diff)
    index = int(np.argmin(np.abs(fnr - fmr)))
    return float((fnr[index] + fmr[index]) / 2.0)


@dataclass(frozen=True)
class Transform:
    """一次线性投影 + 打分度量。LDA 输出是坐标，必须用欧氏距离。"""
    mean: np.ndarray
    weights: np.ndarray
    normalize: bool = True

    @property
    def metric(self):
        return "cosine" if self.normalize else "euclid"

    def __call__(self, X):
        out = (X - self.mean) @ self.weights
        return l2norm(out) if self.normalize else out


@dataclass(frozen=True)
class Composed:
    first: "Transform"
    second: "Transform"

    @property
    def metric(self):
        return self.second.metric

    def __call__(self, X):
        return self.second(self.first(X))


def pca_whiten(X, k=80):
    """白化（无监督），返回 Transform，打分走余弦。"""
    mean = X.mean(0)
    _, singular, vt = np.linalg.svd(X - mean, full_matrices=False)
    k = int(min(k, len(singular)))
    return Transform(mean, vt[:k].T / (singular[:k] / np.sqrt(len(X)) + 1e-9))


def lda_fit(X, y, shrink=0.05, n_components=None, min_class_windows=2):
    """带收缩的 LDA。d=192 常大于样本数，不收缩则类内散度奇异。

    两类场景（within/group 腿）只有 1 个判别方向；LDA 输出是**坐标**不是方向，
    一维上做余弦恒为 ±1、没有区分度，所以判分必须走欧氏距离（normalize=False）。
    """
    classes = [c for c in np.unique(y) if (y == c).sum() >= min_class_windows]
    if len(classes) < 2:
        return None
    d = X.shape[1]
    mean = X.mean(0)
    sw = np.zeros((d, d))
    sb = np.zeros((d, d))
    for c in classes:
        Xc = X[y == c]
        mc = Xc.mean(0)
        delta = Xc - mc
        sw += delta.T @ delta
        diff = (mc - mean).reshape(-1, 1)
        sb += len(Xc) * (diff @ diff.T)
    sw /= len(X)
    sb /= len(X)
    sw += shrink * np.trace(sw) / d * np.eye(d)
    values, vectors = np.linalg.eigh(sw)
    whitener = vectors / np.sqrt(np.maximum(values, 1e-9))
    sb_w = whitener.T @ sb @ whitener
    values2, vectors2 = np.linalg.eigh(sb_w)
    keep = int(n_components or min(len(classes) - 1, 64))
    order = np.argsort(-values2)[:keep]
    return Transform(mean, whitener @ vectors2[:, order], normalize=False)


def pca_lda_fit(X, y, k=60, shrink=0.05, n_components=None):
    """PCA 降维后再 LDA，返回复合 Transform（欧氏距离）。"""
    first = pca_whiten(X, k)
    second = lda_fit(first(X), y, shrink=shrink, n_components=n_components)
    return None if second is None else Composed(first, second)


def pair_scores(Va, Vb, ta, tb, gap, metric="cosine", cap=300, seed=0):
    """同人分数（同一个说话人、两窗不重叠）与异人分数（跨这两个说话人）。

    `metric="euclid"` 返回**负**平方距离，维持"分数越高越像同一人"的约定。
    """
    rng = np.random.default_rng(seed)

    def take(V, t):
        index = rng.choice(len(V), min(len(V), cap), replace=False)
        return V[index], t[index]

    Va, ta = take(Va, ta)
    Vb, tb = take(Vb, tb)
    def similarity(A, B):
        if metric == "euclid":
            return -((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return A @ B.T

    same = []
    for V, t in ((Va, ta), (Vb, tb)):
        grams = similarity(V, V)
        gap_ok = np.abs(t[:, None] - t[None, :]) >= gap
        upper = np.triu(np.ones_like(grams, dtype=bool), k=1)
        same.append(grams[gap_ok & upper])
    same = np.concatenate([x for x in same if len(x)]) if any(len(x) for x in same) else np.array([])
    diff = similarity(Va, Vb).ravel()
    return same, diff


def split_half(times):
    """按时间劈一半（前一半拟合，后一半回评），避免同窗既训又测。"""
    order = np.argsort(times)
    half = len(order) // 2
    return order[:half], order[half:]


def score_pair(Va, Vb, ta, tb, gap, metric="cosine"):
    same, diff = pair_scores(Va, Vb, ta, tb, gap, metric=metric)
    value = eer(same, diff)
    if value is None:
        return None, None, len(same), len(diff)
    interval = (float(np.median(diff)) - float(np.median(same))
                if len(same) and len(diff) else None)
    return value, interval, len(same), len(diff)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\yinp\AISHELL-4\data\test")
    ap.add_argument("--cache", default=r"D:\yinp\AISHELL-4\_v13_spike\cache")
    ap.add_argument("--meeting")
    ap.add_argument("--split", choices=("dev", "holdout"), default="dev")
    ap.add_argument("--seconds", type=float, default=600.0)
    ap.add_argument("--window", type=float, default=2.0)
    ap.add_argument("--hop", type=float, default=1.0)
    ap.add_argument("--gap", type=float, default=2.0, help="同人窗对的最小时间间隔（秒）")
    ap.add_argument("--min-windows", type=int, default=4, help="一对说话人各自的最少纯窗数")
    ap.add_argument("--output")
    args = ap.parse_args()

    root = Path(args.root)
    meetings = [args.meeting] if args.meeting else split_meetings(root, args.split)

    pool = {}
    for meeting in meetings:
        emb, times, speakers, votes = load(args.cache, meeting, args.window, args.hop)
        pool[meeting] = (meeting_group(meeting),
                         *pure_windows(emb, times, speakers, votes))

    rows = []
    for meeting in meetings:
        group, emb, times, speakers = pool[meeting]
        by_speaker = {}
        for name in sorted(set(speakers.tolist())):
            mask = speakers == name
            if mask.sum() >= args.min_windows:
                by_speaker[name] = (emb[mask], times[mask])
        if len(by_speaker) < 2:
            continue

        # 该场"最像的一对"（用质心余弦，与上一轮探针同法）
        names = sorted(by_speaker)
        centroids = np.stack([l2norm(by_speaker[n][0].mean(0)) for n in names])
        similarity = centroids @ centroids.T
        np.fill_diagonal(similarity, -1.0)
        i, j = np.unravel_index(int(similarity.argmax()), similarity.shape)
        a, b = names[int(i)], names[int(j)]
        Va, ta = by_speaker[a]
        Vb, tb = by_speaker[b]

        row = {"meeting": meeting, "group": group, "pair": [a, b],
               "n_a": int(len(Va)), "n_b": int(len(Vb)),
               "centroid_cosine": round(float(similarity[i, j]), 4)}

        raw, interval, n_same, n_diff = score_pair(l2norm(Va), l2norm(Vb), ta, tb, args.gap)
        row["raw"] = {"eer": raw, "median_gap": interval,
                      "n_same": n_same, "n_diff": n_diff}

        def apply_and_score(fit, Va, Vb):
            if fit is None:
                return {"eer": None}
            value, gapd, _, _ = score_pair(fit(Va), fit(Vb), ta, tb, args.gap,
                                           metric=fit.metric)
            return {"eer": value, "median_gap": gapd}

        # within：目标场自己劈一半拟合，另一半回评（信息在不在的上限）
        train_index, test_index = split_half(np.concatenate([ta, tb]))
        n_a = len(ta)
        tr_a = train_index[train_index < n_a]
        tr_b = train_index[train_index >= n_a] - n_a
        te_a = test_index[test_index < n_a]
        te_b = test_index[test_index >= n_a] - n_a
        if min(len(tr_a), len(tr_b), len(te_a), len(te_b)) >= 2:
            X = np.concatenate([Va[tr_a], Vb[tr_b]])
            y = np.array([a] * len(tr_a) + [b] * len(tr_b))
            row["within_pca_lda"] = apply_and_score(pca_lda_fit(X, y), Va[te_a], Vb[te_b])
            row["within_lda"] = apply_and_score(lda_fit(X, y), Va[te_a], Vb[te_b])

        # group：同一身份组其它会议里的**同一对**说话人
        other = []
        for meeting2 in meetings:
            if meeting2 == meeting:
                continue
            group2, emb2, times2, spk2 = pool[meeting2]
            if group2 != group:
                continue
            for name, target in ((a, "a"), (b, "b")):
                mask = spk2 == name
                if mask.sum() >= 2:
                    other.append((emb2[mask], name))
        if len({n for _, n in other}) == 2:
            X = np.concatenate([v for v, _ in other])
            # 每个**窗**一个标签：按窗数展开，不能按说话人数（否则 y 长度与 X 不匹配）
            y = np.concatenate([np.full(len(v), n) for v, n in other])
            row["group_pca_lda"] = apply_and_score(pca_lda_fit(X, y), Va, Vb)
            row["group_lda"] = apply_and_score(lda_fit(X, y), Va, Vb)
            row["group_whiten"] = apply_and_score(pca_whiten(X, 80), Va, Vb)
            row["group_fit_windows"] = int(len(X))

        # global：其它会议的**全部**说话人（可部署的后端，不挑人）
        others = []
        for meeting2 in meetings:
            if meeting2 == meeting:
                continue
            _, emb2, times2, spk2 = pool[meeting2]
            for name in sorted(set(spk2.tolist())):
                mask = spk2 == name
                if mask.sum() >= 2:
                    others.append((emb2[mask], name))
        if len({n for _, n in others}) >= 3:
            X = np.concatenate([v for v, _ in others])
            # 每个**窗**一个标签：按窗数展开，不能按说话人数（否则 y 长度与 X 不匹配）
            y = np.concatenate([np.full(len(v), n) for v, n in others])
            row["global_pca_lda"] = apply_and_score(pca_lda_fit(X, y), Va, Vb)
            row["global_lda"] = apply_and_score(lda_fit(X, y), Va, Vb)
            row["global_fit_windows"] = int(len(X))
            row["global_fit_speakers"] = int(len(set(y.tolist())))

        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def cell(row, key):
        value = (row.get(key) or {}).get("eer")
        return f"{value * 100:6.1f}%" if value is not None else "     -"

    print(f"\n{'会议':<16}{'最像一对':>14}{'余弦':>7}{'raw':>8}"
          f"{'within':>8}{'group':>8}{'global':>8}{'wf':>6}{'gf':>6}")
    for row in rows:
        pair = "/".join(row["pair"])
        print(f"{row['meeting']:<16}{pair:>14}{row['centroid_cosine']:>7.3f}"
              f"{cell(row, 'raw'):>8}{cell(row, 'within_pca_lda'):>8}"
              f"{cell(row, 'group_pca_lda'):>8}{cell(row, 'global_pca_lda'):>8}"
              f"{str(row.get('group_fit_windows', '-')):>6}"
              f"{str(row.get('global_fit_windows', '-')):>6}")

    def mean_of(key):
        values = [(r.get(key) or {}).get("eer") for r in rows]
        values = [v for v in values if v is not None]
        return f"{np.mean(values) * 100:.1f}% (n={len(values)})" if values else "-"

    print(f"\n平均 EER: raw={mean_of('raw')}  within_PCA+LDA={mean_of('within_pca_lda')}"
          f"  group_PCA+LDA={mean_of('group_pca_lda')}  global_PCA+LDA={mean_of('global_pca_lda')}")
    print(f"           within_LDA={mean_of('within_lda')}  group_LDA={mean_of('group_lda')}"
          f"  global_LDA={mean_of('global_lda')}  group_whiten={mean_of('group_whiten')}")

    if args.output:
        Path(args.output).write_text(
            json.dumps({"schema": "v13-pair-oracle-v1", "meetings": rows},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n写入 {args.output}")


if __name__ == "__main__":
    main()
