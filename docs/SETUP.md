# Setup in VS Code

## 1. Folder layout (merge Block A files into this)
```
pattern-anomaly/
├── config.yaml                 (Block A)
├── requirements.txt
├── .env  .gitignore  .vscode/
├── data/                       (Block A output: normal/, altered_margin/, inserted_signature/, metadata/annotations.csv)
│   └── features/               (created by: python -m ml.pipeline --build)
├── ml/
│   ├── generate_dataset.py     (Block A)
│   ├── extract.py              (Block A)
│   ├── features.py             (Block A)
│   ├── common.py regions.py scoring.py pipeline.py train.py evaluate.py   (this block)
├── models/                     (created by train: scaler+model .pkl, feature_config.json)
├── reports/                    (created by evaluate)
├── backend/app/                main.py, api/routes.py, services/, schemas.py, errors.py
└── tests/
```

## 2. Environment
```
python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```
Install Tesseract (only needed for scanned PDFs) and make sure `tesseract` is on PATH.
In VS Code: Ctrl+Shift+P -> "Python: Select Interpreter" -> .venv.

## 3. Connect Block A (one file only)
Open `ml/pipeline.py`, edit `extract_page_records()` so it calls your real functions in
extract.py / features.py. Then verify:
```
python -m ml.pipeline --check data/normal/doc_001.pdf
```

## 4. Run the ML pipeline
```
python -m ml.pipeline --build      # features + elements tables
python -m ml.train                 # grouped split, scaler+IsolationForest on train-normal, 95th-pct threshold
python -m ml.evaluate --compare    # metrics, confusion matrix, IoU, experiment table -> reports/
```
(Or Terminal > Run Task > "ML: full pipeline".)

## 5. Run the backend
```
python -m uvicorn backend.app.main:app --reload --port 8000
```
Docs at http://localhost:8000/docs. Endpoints: POST /api/analyze, GET /api/analysis/{id},
/page/{n}, /features, /visualization.

## 6. Tests
```
python -m pytest -q
```
