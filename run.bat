@echo off
REM ---------------------------------------------------------------
REM  JobRec - AI-Powered Job Recommendation System (Windows launcher)
REM  Double-click this file. It installs requirements, trains the
REM  model (first run only), runs the evaluation and starts the app.
REM ---------------------------------------------------------------
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.9+ from https://www.python.org/downloads/ and tick "Add python to PATH".
  pause
  exit /b 1
)
echo [1/4] Installing requirements ...
python -m pip install -q -r requirements.txt
if not exist models\hybrid_recommender.pkl (
  echo [2/4] Training the recommender (about 15 seconds) ...
  python train.py
) else (
  echo [2/4] Model already trained - skipping. Delete the models folder to retrain.
)
if not exist outputs\evaluation_results.json (
  echo [3/4] Running offline evaluation (about 30 seconds) ...
  python evaluate.py
) else (
  echo [3/4] Evaluation results already exist - skipping.
)
echo [4/4] Starting the web app ...
start "" http://127.0.0.1:5000
python app\app.py
pause
