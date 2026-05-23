# 📧 Email Spam Detection using Deep Learning

A deep learning project that detects spam emails using LSTM and Bidirectional LSTM models. The project compares four model variants — training from scratch vs. using pre-trained GloVe word embeddings — and evaluates them using multiple metrics.

---

## 📁 Dataset

- **File:** `Emails.csv`
- **Columns:** `text` (email content), `spam` (label: 1 = spam, 0 = not spam)
- **Note:** The dataset is imbalanced, which is handled using class weights during training.

---

## 🔧 Project Pipeline

1. **Load & Explore Data** — preview with `head()`, `tail()`, `describe()`
2. **Label Encoding** — convert labels to binary (0 / 1)
3. **Tokenization & Padding** — vocabulary size: 1000, max sequence length: 100
4. **Train / Val / Test Split** — 80% train, 10% validation, 10% test
5. **Class Imbalance Handling** — balanced class weights via `compute_class_weight`
6. **GloVe Embeddings** — loaded from `glove.6B.100d.txt` (100-dimensional)
7. **Model Training** — 4 models trained with EarlyStopping (patience = 3)
8. **Evaluation** — Accuracy, F1-Score, ROC-AUC, Confusion Matrix

---

## 🧠 Models

| Model | Architecture | Embeddings |
|-------|-------------|------------|
| **A_scratch** | Embedding → LSTM(64) → Dense(sigmoid) | Random (trained from scratch) |
| **A_Transfer** | Embedding → LSTM(64) → Dense(sigmoid) | GloVe 100d (frozen) |
| **B_scratch** | Embedding → BiLSTM(64) → Dropout(0.5) → Dense(sigmoid) | Random (trained from scratch) |
| **B_transfer** | Embedding → BiLSTM(64) → Dropout(0.5) → Dense(sigmoid) | GloVe 100d (frozen) |

---

## 📊 Evaluation Metrics

Each model is evaluated on the test set using:
- **Accuracy**
- **F1-Score**
- **ROC-AUC Score**
- **Confusion Matrix**

---

##  How to Run

1. Open the notebook in **Google Colab**
2. Upload `archive (6).zip` (contains `Emails.csv`) to your Google Drive
3. Download GloVe embeddings:
```bash
   !wget https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
   !unzip -q glove.6B.zip
```
4. Mount your Google Drive and update the dataset path if needed
5. Run all cells in order

---

##  Requirements

- Python 3.x
- TensorFlow / Keras
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn
- Google Colab (recommended)

---

##  Repository Structure

```
├── DeepPR2.ipynb          # Main notebook
├── archive (6).zip        # Dataset (Emails.csv)
└── README.md
```
