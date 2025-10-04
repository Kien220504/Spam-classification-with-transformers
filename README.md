# Spam-classification-with-transformers

Spam classification with **Transformers)** — fine-tuning **DistilBERT `distilbert-base-uncased`** on an SMS/Email dataset, then **deploying** an inference UI with **Gradio**.

---

## 1) Project structure

```
Spam-classification-with-transformers
├── datasets
│   ├── spam.csv
│   ├── train.csv        # created by running notebooks/data_prepare.ipynb
│   └── test.csv         # created by running notebooks/data_prepare.ipynb
├── notebooks
│   ├── data_prepare.ipynb  # preprocess raw data and save train/test
│   └── evaluate.ipynb      # evaluate the trained model
├── scripts
│   ├── app.py           # Gradio deployment
│   └── train.py         # model training pipeline
├── requirements.txt
└── README.md
```

**Path note:**
All notebooks/scripts assume the relative paths above. If you move folders/files, **update paths** in the notebooks (`notebooks/*.ipynb`) and scripts (`scripts/*.py`) accordingly (e.g., `datasets/train.csv`, `models/...`).

---

## 2) Setup & installation

* **Python** ≥ 3.10
* **PyTorch**: match your CPU/GPU/CUDA setup

```bash
# (Recommended) create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install all dependencies from requirements.txt
pip install -r requirements.txt
```

> 💡 If your machine needs a specific **PyTorch** build (CPU/GPU/CUDA), adjust the `torch` line in `requirements.txt` accordingly.

---

## 3) Data

* `datasets/spam.csv`: raw data. Include **text** (column *v2*) and **label** (column *v1*).
* Run `notebooks/data_prepare.ipynb` to:

  * read `spam.csv`, clean/split data
  * write `datasets/train.csv` and `datasets/test.csv`

**Path note:** If `spam.csv` is not under `datasets/`, edit the path inside `notebooks/data_prepare.ipynb` before running.

---

## 4) Training

Script: `scripts/train.py`

Typical usage:

```bash
python scripts/train.py
```

**Path/column notes:**

* If your CSV column names differ or files live elsewhere, **adjust the arguments**.
* `output_dir` is where the fine-tuned model is saved.

**✅ After training completes, the fine-tuned model artifacts are saved under the repository’s `models/` folder** (e.g., `./models/distilbert-base-uncased`).

The model directory usually contains:

```
config.json
pytorch_model.bin
tokenizer.json
tokenizer_config.json
special_tokens_map.json
...
```

---

## 5) Evaluation

Notebook: `notebooks/evaluate.ipynb`

* Load the fine-tuned model from `models/...`
* Evaluate on `datasets/test.csv` (accuracy, precision, recall, F1, confusion matrix)

**Path note:** Set correct paths to the model dir and `datasets/test.csv` inside the notebook.

---

## 6) Deploy the UI with Gradio

Script: `scripts/app.py`

* Loads the model/tokenizer from `models/...`
* Launches a **Gradio Interface** to input text and return Spam/Ham

**Local run:**

```bash
python scripts/app.py
```

**Server run (Linux, external access):**

* Ensure in `app.py` something like:

  ```python
  demo.launch(
      server_name='0.0.0.0',
      server_port=int(os.getenv("PORT", "7860")),
      inbrowser=False
  )
  ```
  
* Open the firewall/NAT for the chosen port or set `share=True` in `demo.launch()`.

**Path note:** If `app.py` uses a constant like `MODEL_PATH = "models/distilbert-base-uncased"`, update it to match your trained model directory (absolute or relative path as needed).

---

## 7) Quickstart

1. `pip install -r requirements.txt`
2. Run `notebooks/data_prepare.ipynb` → creates `datasets/train.csv` & `datasets/test.csv`
3. **Train:** `python scripts/train.py ...`

   * **Verify the project’s `./models/` folder now contains your trained model** (e.g., `./models/distilbert-base-uncased`).
4. Evaluate: open `notebooks/evaluate.ipynb`
5. Deploy: `python scripts/app.py` -> open `http://localhost:7860` (or your configured host/port)

---

## 8) Tips & troubleshooting

* **No CUDA/GPU:** CPU works but slower; pick the right **PyTorch build** for your hardware.
* **OOM** reduce `batch_size`, shorten `max_seq_length` if exposed.
* **Model not found when deploying:** double-check the **model directory path** in `app.py` or CLI args.

---

## 9) Extensibility

* Swap base model: e.g., `bert-base-uncased`, `roberta-base` via `MODEL_NAME` (from `train.py`).
* Optional: Dockerize for stable deployment; add logging/MLflow/Weights & Biases as needed.
