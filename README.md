Relay Capacity Tier Classification

Benchmarking Classical Machine Learning Models Against a Simple Neural Network
📌 Overview

This project presents a supervised machine learning benchmark for classifying Tor network relays into capacity tiers using self-collected relay metadata.

The goal is to evaluate and compare the performance of six classical machine learning models against a Simple Neural Network (SNN) on a real-world, non-benchmark dataset.

    Important:
    This project operates strictly on public, infrastructure-level metadata.
    No user traffic, payloads, circuits, or deanonymization techniques are involved.

🎯 Problem Statement

Given a set of observable Tor relay metadata, can we accurately classify each relay into a capacity tier—Low, Medium, or High—based on its operational characteristics?

Formally:

    Input: Relay-level operational metadata
    Output: Capacity tier ∈ {Low, Medium, High}

This is framed as a multi-class supervised classification problem.
🧠 Motivation

    Tor relays differ significantly in capacity and trust

    Manual rule-based classification is brittle and non-scalable

    Capacity is influenced by multiple interacting factors

    Machine learning enables generalized, data-driven classification

This study also serves as a comparative benchmark, highlighting where:

    Classical ML models excel on structured data

    Neural networks perform competitively but not dominantly

📂 Dataset Description
Data Source

    Relay metadata collected locally from a Tor relay monitoring system

    Stored originally in a relational database

    Exported using read-only SQL queries

    No pre-existing ML datasets (e.g., Kaggle, UCI) were used

Data Ownership

    Fully self-collected

    Primary data

    Experimentally derived labels

Features Used
Feature	Description
bandwidth	Advertised relay bandwidth
consensus_weight	Trust-weighted capacity assigned by Tor
orport	Onion routing port
is_exit	Exit relay flag (binary)
is_guard	Guard relay flag (binary)
is_fast	Fast relay flag (binary)
is_stable	Stable relay flag (binary)
tor_major	Major Tor version number

All features are numeric and suitable for ML models.
Target Variable — Capacity Tier

Capacity tiers are defined explicitly using consensus_weight:
Tier	Label	Definition
Low	0	consensus_weight < 1,000
Medium	1	1,000 ≤ consensus_weight ≤ 10,000
High	2	consensus_weight > 10,000

This labeling logic is original, deterministic, and reproducible.
Class Distribution

High (2)   : 5738
Medium (1): 3586
Low (0)   : 3500

The dataset is well-balanced, requiring no resampling or synthetic augmentation.
🔬 Experimental Methodology
1. Data Preprocessing

    Feature/label separation

    Stratified train–test split (80/20)

    Feature standardization using StandardScaler

2. Models Evaluated (6 + 1 Benchmark)
Classical Machine Learning Models

    Logistic Regression

    Naive Bayes

    K-Nearest Neighbors (KNN)

    Support Vector Machine (SVM)

    Random Forest

    XGBoost

Neural Model

    Simple Neural Network (SNN)

3. Simple Neural Network Architecture

    Fully connected feed-forward network

    Two hidden layers with ReLU activation

    Softmax output layer (3 classes)

    Trained using categorical cross-entropy

This network is intentionally simple, serving as a fair comparator rather than an over-engineered deep model.
4. Evaluation Metrics

    Accuracy (primary metric)

    Confusion Matrix (error analysis)

All models are evaluated on the same test set for fairness.
📊 Results & Observations

Key trends observed:

    Tree-based ensemble models (Random Forest, XGBoost) achieved the highest accuracy

    SVM performed competitively

    Simple Neural Network demonstrated strong performance but did not outperform gradient-boosted trees

    Linear and probabilistic models showed lower performance due to limited capacity to model non-linear feature interactions

This behavior aligns with established theory for tabular structured data.
🧾 Confusion Matrix Analysis

Confusion matrices reveal that:

    Most misclassifications occur between adjacent tiers (e.g., Medium ↔ High)

    Extreme misclassifications (Low ↔ High) are rare

    The model captures capacity trends but struggles at boundary cases

🛠️ Project Structure

.
├── relay_capacity_dataset.csv
├── relay_capacity_benchmark.ipynb
├── README.md

    relay_capacity_dataset.csv → Final ML dataset

    relay_capacity_benchmark.ipynb → Full experiment notebook

    README.md → Project documentation

⚠️ Ethical Considerations

    No user data is collected or analyzed

    No traffic payloads or circuits are inspected

    No deanonymization or surveillance techniques are used

    All data is public, relay-level metadata

This project complies with ethical research practices in network measurement.
🧪 Reproducibility

To reproduce the experiment:

    Clone the repository

    Install dependencies

    Run the notebook top-to-bottom

Dependencies

    Python 3.11+

    pandas

    numpy

    scikit-learn

    xgboost

    tensorflow

    matplotlib

📌 Key Takeaways

    Classical ML models remain highly effective on structured tabular data

    Neural networks are not universally superior

    Model selection must consider data type and structure

    Benchmarking provides insight beyond raw accuracy

📄 License

This project is released for academic and educational use.
