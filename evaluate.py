"""
evaluate.py - offline evaluation of the recommender (precision@k, recall@k, hit-rate, NDCG)

Writes:
  outputs/evaluation_results.json
  outputs/evaluation_report.md
  outputs/precision_recall_at_k.png
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from jobrec import ContentModel, load_jobs  # noqa: E402
from jobrec.evaluate import evaluate  # noqa: E402
from jobrec.simulate import generate_users  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "outputs")


def main(n_users=300, alpha=0.3):
    os.makedirs(OUT, exist_ok=True)
    jobs = load_jobs().reset_index(drop=True)
    content = ContentModel().fit(jobs)
    users, interactions, _ = generate_users(jobs, content, n_users=n_users)
    print(f"Evaluating on {len(users)} users / {len(interactions):,} interactions ...")
    summary = evaluate(jobs, users, interactions, ks=(5, 10, 20), alpha=alpha)

    with open(os.path.join(OUT, "evaluation_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    models = ["popularity", "content", "collaborative", "hybrid"]
    lines = ["# Offline evaluation report", "",
             f"Users: {summary['_meta']['n_users']} | Jobs: {summary['_meta']['n_jobs']:,} | "
             f"Train interactions: {summary['_meta']['n_train_interactions']:,} | "
             f"Test interactions: {summary['_meta']['n_test_interactions']:,} | alpha = {alpha}", "",
             "| Model | k | Precision@k | Recall@k | Hit-rate@k | NDCG@k |", "|---|---|---|---|---|---|"]
    for m in models:
        for k in (5, 10, 20):
            r = summary[m][k]
            lines.append(f"| {m} | {k} | {r['precision']:.3f} | {r['recall']:.3f} | {r['hit_rate']:.3f} | {r['ndcg']:.3f} |")
    lines += ["", "## Hybrid weight (alpha) sweep - k = 10", "",
              "alpha = 1.0 is pure content-based, alpha = 0.0 is pure collaborative.", "",
              "| alpha | Precision@10 | Recall@10 | Hit-rate@10 | NDCG@10 |", "|---|---|---|---|---|"]
    for name in summary:
        if name.startswith("hybrid_alpha="):
            r = summary[name][10]
            lines.append(f"| {name.split('=')[1]} | {r['precision']:.3f} | {r['recall']:.3f} | {r['hit_rate']:.3f} | {r['ndcg']:.3f} |")
    report = "\n".join(lines)
    with open(os.path.join(OUT, "evaluation_report.md"), "w") as f:
        f.write(report + "\n")
    print(report)

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ks = [5, 10, 20]
    for m in models:
        axes[0].plot(ks, [summary[m][k]["precision"] for k in ks], marker="o", label=m)
        axes[1].plot(ks, [summary[m][k]["recall"] for k in ks], marker="o", label=m)
    axes[0].set_title("Precision@k"); axes[1].set_title("Recall@k")
    for ax in axes:
        ax.set_xlabel("k"); ax.set_xticks(ks); ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "precision_recall_at_k.png"), dpi=130)
    print("saved plots to", OUT)
    return summary


if __name__ == "__main__":
    main()
