---

````markdown
# Relay Capacity Tier Classification  
### Benchmarking Classical Machine Learning Models vs a Simple Neural Network

---

## Overview

This repository contains a **complete machine learning benchmarking study** focused on **classifying Tor network relays into capacity tiers** using **self-collected relay metadata**.

> **No pre-used ML datasets**  
> **No user traffic or deanonymization**  
> **Only infrastructure-level, public relay metadata**

The project compares **6 classical machine learning models** against **1 Simple Neural Network (SNN)** using a **single, reproducible notebook**.

---

## Objective

> **Predict the capacity tier of a Tor relay**  
> *(Low / Medium / High)*  
> based solely on its **operational metadata**.

This is framed as a **multi-class supervised classification problem**.

---

## Why This Matters

- Tor relays vary significantly in capacity and trust
- Manual or rule-based classification is brittle
- Capacity depends on **multiple interacting features**
- Machine learning enables **generalized, data-driven inference**

This project also demonstrates that:

> **Neural networks are not always superior on structured tabular data**

---

## Dataset

### Source
- Self-collected relay metadata
- Extracted from a local database using **read-only SQL**
- No Kaggle / UCI / benchmark datasets used

### Nature of Data
- Primary data
- Real-world, noisy
- Not curated for ML originally

---

## 🧾 Features Used

| Feature | Description |
|------|------------|
| `bandwidth` | Advertised relay bandwidth |
| `consensus_weight` | Trust-weighted capacity |
| `orport` | Onion routing port |
| `is_exit` | Exit relay flag |
| `is_guard` | Guard relay flag |
| `is_fast` | Fast relay flag |
| `is_stable` | Stable relay flag |
| `tor_major` | Major Tor version |

✔ All features are numeric  
✔ Identifiers (IP, fingerprint) removed  

---

## Capacity Tier Labels

Capacity tiers are **explicitly defined** using `consensus_weight`:

```text
Low    (0): < 1,000
Medium (1): 1,000 – 10,000
High   (2): > 10,000
````
This labeling logic is **original and reproducible**.

---

## Class Distribution

```text
High (2)   : 5738
Medium (1): 3586
Low (0)   : 3500
```

✔ Well-balanced
✔ No resampling required

---

## Methodology

### Preprocessing

* Feature / label separation
* Stratified train–test split (80 / 20)
* Feature scaling using `StandardScaler`

---

### Models Evaluated (6 + 1)

#### Classical ML Models

1. Logistic Regression
2. Naive Bayes
3. K-Nearest Neighbors (KNN)
4. Support Vector Machine (SVM)
5. Random Forest
6. XGBoost

#### ➕ Neural Model

7. **Simple Neural Network (SNN)**

---

##  Simple Neural Network (SNN)

* Fully connected feed-forward network
* Two hidden layers (ReLU)
* Softmax output layer (3 classes)
* Loss: categorical cross-entropy
* Optimizer: Adam

 The network is **intentionally simple** to ensure a **fair benchmark**.

---

##  Evaluation Metrics

* Accuracy (primary)
* Confusion Matrix (error analysis)

All models are evaluated on the **same test set**.

---

## 📉 Key Observations

✔ Tree-based ensemble models (Random Forest, XGBoost) performed best
✔ SVM showed strong competitive performance
✔ SNN was competitive but did not outperform boosted trees
✔ Linear & probabilistic models underperformed on non-linear patterns

📌 These results align with established ML theory for tabular data.

---

## 🧩 Confusion Matrix Insights

* Most errors occur between **adjacent tiers**
* Very few extreme misclassifications (Low ↔ High)
* Indicates strong trend learning with boundary ambiguity

---

## 🗂️ Repository Structure

```text
.
├── relay_capacity_dataset.csv
├── relay_capacity_benchmark.ipynb
├── README.md
```

---

##  Reproducibility

To reproduce:

```bash
git clone <repo>
cd <repo>
jupyter notebook
```

Run the notebook **top to bottom**.

### 📦 Dependencies

* Python 3.11+
* pandas
* numpy
* scikit-learn
* xgboost
* tensorflow
* matplotlib

---

## 📌 Takeaways

> 🔹 Model choice must match data structure
> 🔹 Neural networks are not universally superior
> 🔹 Benchmarking reveals strengths and limitations clearly

---

## ✍️ Author

Developed as part of a **cybersecurity-aligned machine learning study**
focused on **network infrastructure analytics and model benchmarking**.

---

⭐ *If you found this project useful, consider starring the repository.*

```

---

