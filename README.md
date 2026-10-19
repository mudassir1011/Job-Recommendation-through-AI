# JobRec – AI-Powered Job Recommendation System
### Using Machine Learning for Skills-Based Career Matching

**Student:** Syed Mohiuddin Mudassir (415789) · **Course:** MSc Computer Science · **Supervisor:** Dr. Paul Sant

A web-based intelligent job recommendation engine that

1. parses user profiles (skills, education, experience),
2. processes job postings with NLP (TF-IDF),
3. generates personalised recommendations with a **hybrid** of content-based and collaborative filtering, and
4. evaluates accuracy with **precision@k / recall@k** (plus hit-rate@k and NDCG@k).

---

## 1. Quick start

### Option A – Google Colab (no installation)
1. Open `JobRec_Colab.ipynb` in Google Colab (File → Upload notebook).
2. Run the cells top to bottom. The first cell asks you to upload `jobrec.zip`.
3. The last cell prints a link – click it to open the running web app.

### Option B – Windows
1. Install Python 3.9+ from python.org (tick **"Add python to PATH"**).
2. Unzip `jobrec.zip`, open the folder and double-click **`run.bat`**.
3. Your browser opens at http://127.0.0.1:5000. Log in with **demo / demo123**.

### Option C – macOS / Linux
```bash
unzip jobrec.zip && cd jobrec
bash run.sh
```

### Manual steps (any OS)
```bash
pip install -r requirements.txt
python train.py        # builds models/hybrid_recommender.pkl  (~15 s)
python evaluate.py     # writes outputs/evaluation_report.md + plots (~30 s)
python app/app.py      # http://127.0.0.1:5000
```

---

## 2. Project structure

```
jobrec/
├── data/
│   ├── jobs_dataset.csv       curated dataset – 8,000 postings, 127 skills (ships with the project)
│   └── prepare_dataset.py     rebuilds the CSV from the raw Kaggle files (optional)
├── jobrec/                    the recommender package
│   ├── data.py                loading + skill normalisation ("Power BI" -> power_bi)
│   ├── content_model.py       TF-IDF + cosine similarity + skill-set overlap
│   ├── collab_model.py        item-based k-NN + TruncatedSVD matrix factorisation
│   ├── hybrid.py              weighted blend, filters, explanations, save/load
│   ├── simulate.py            mock users & interactions (K-Means "career tracks")
│   └── evaluate.py            precision@k, recall@k, hit-rate@k, NDCG@k
├── app/
│   ├── app.py                 Flask web application (auth, profile, search, recommendations, API)
│   ├── templates/             HTML pages
│   └── static/style.css
├── train.py                   builds and saves the model
├── evaluate.py                offline evaluation -> outputs/
├── outputs/                   evaluation_report.md, evaluation_results.json, plots, screenshots
├── JobRec_Colab.ipynb         one-click Colab notebook
├── run.bat / run.sh           launchers
└── requirements.txt
```

---

## 3. Dataset

Public Kaggle dataset **"Data Analyst Job Postings [Pay, Skills, Benefits]"** by Luke Barousse
(https://www.kaggle.com/datasets/lukebarousse/data-analyst-job-postings-google-search), collected daily from
Google Jobs. Each posting has title, company, location, description, schedule, remote flag, salary (when stated)
and an annotated list of skills (`description_tokens`).

`data/prepare_dataset.py` merged the raw quarterly files (28,375 rows), removed duplicates (20,446), kept postings
with at least two annotated skills (13,142) and sampled 8,000 for a fast, reproducible demo. No personal data is
included – only public postings, in line with the ethics section of the proposal.

---

## 4. How the recommender works

| Stage | Technique | Where |
|---|---|---|
| Profile parsing | tokenising / alias mapping of user skills; free-text experience | `jobrec/data.py` |
| Job representation | TF-IDF (1–2-grams, 40 k features, sub-linear tf, stop-words) over title ×3 + skills ×2 + description; binary skill matrix | `content_model.py` |
| Content score | ½ · cosine(TF-IDF profile, job) + ½ · cosine(skill sets) | `content_model.py` |
| Collaborative score | interactions (view 1 / save 3 / apply 5, log-dampened) → item-item cosine k-NN **and** TruncatedSVD (40 latent factors); averaged | `collab_model.py` |
| Hybrid | `score = α·content + (1−α)·CF`, α = 0.3 (grid-searched); cold-start → content only | `hybrid.py` |
| Filters | location, remote-only, minimum salary | `hybrid.py` |
| Explanation | matched vs. missing skills for every result | `hybrid.py` + templates |
| Online learning | CF model refits automatically every 5 new interactions (or "Re-train CF now") | `app/app.py` |

---

## 5. Evaluation results (offline, 300 mock users, 70/30 split)

| Model | Precision@10 | Recall@10 | Hit-rate@10 | NDCG@10 |
|---|---|---|---|---|
| popularity baseline | 0.011 | 0.026 | 0.077 | 0.020 |
| content-based | 0.018 | 0.050 | 0.137 | 0.038 |
| collaborative | 0.076 | 0.194 | 0.430 | 0.154 |
| **hybrid (α = 0.3)** | **0.089** | **0.226** | **0.480** | **0.182** |

Random chance for precision@10 is ≈ 0.0005 (≈ 4 relevant jobs among 8,000). The hybrid beats popularity ~8× and
both single models, confirming the hypothesis that combining content and collaborative signals improves matching.
Full tables (k = 5/10/20 and the α sweep) are in `outputs/evaluation_report.md` and on the app's *Evaluation* page.

---

## 6. Web application pages

| URL | Purpose |
|---|---|
| `/` | landing page, dataset statistics |
| `/register`, `/login` | authentication (hashed passwords, SQLite) |
| `/profile` | skills, target role, experience text, filters |
| `/recommendations` | personalised top-k with scores, matched/missing skills, Save/Apply |
| `/jobs?q=` | classic keyword search for comparison |
| `/job/<id>` | posting detail + similar jobs; records a *view* |
| `/saved` | interaction history |
| `/evaluation` | precision/recall tables and plots |
| `/about` | architecture explanation |
| `/api/recommend?skills=python,sql&title=Data%20Analyst&k=5` | JSON REST API |

Demo account: **demo / demo123** (pre-filled profile with some history so the *hybrid* path is exercised).

---

## 7. References
- Resnick, P. & Varian, H.R. (1997). Recommender systems. *Communications of the ACM*, 40(3).
- Koren, Y., Bell, R. & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. *IEEE Computer*, 42(8).
- Covington, P., Adams, J. & Sargin, E. (2016). Deep neural networks for YouTube recommendations. *RecSys '16*.
- Salton, G. & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management*.
