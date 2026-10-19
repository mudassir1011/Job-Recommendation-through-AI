"""
train.py - build the recommender and save it to models/hybrid_recommender.pkl

Steps
  1. load the curated dataset (data/jobs_dataset.csv)
  2. generate mock users + interactions (seeded, reproducible)
  3. fit the content model (TF-IDF) and the collaborative model (item-kNN + SVD)
  4. store the mock users so the web app can pre-populate demo accounts
"""
import json
import os
import time

from jobrec import HybridRecommender, load_jobs
from jobrec.simulate import generate_users

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "models")


def main(n_users=300):
    t0 = time.time()
    jobs = load_jobs()
    print(f"[1/4] loaded {len(jobs):,} job postings, {jobs['skill_list'].explode().nunique()} unique skills")

    rec = HybridRecommender(alpha=0.3)
    rec.content.fit(jobs.reset_index(drop=True))
    users, interactions, tracks = generate_users(jobs, rec.content, n_users=n_users)
    print(f"[2/4] generated {len(users)} mock users and {len(interactions):,} interactions")

    rec.fit(jobs, interactions)
    print(f"[3/4] fitted hybrid model: TF-IDF matrix {rec.content.job_matrix.shape}, "
          f"CF matrix {rec.collab.R.shape}, {rec.collab.item_factors.shape[1]} latent factors")

    os.makedirs(MODEL_DIR, exist_ok=True)
    path = rec.save()
    with open(os.path.join(MODEL_DIR, "mock_users.json"), "w") as f:
        json.dump({"users": users, "interactions": interactions}, f)
    print(f"[4/4] saved model -> {path}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
