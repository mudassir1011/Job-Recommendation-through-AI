#!/usr/bin/env bash
# JobRec launcher for macOS / Linux
cd "$(dirname "$0")"
python3 -m pip install -q -r requirements.txt
[ -f models/hybrid_recommender.pkl ] || python3 train.py
[ -f outputs/evaluation_results.json ] || python3 evaluate.py
echo "Open http://127.0.0.1:5000  (demo login: demo / demo123)"
python3 app/app.py
