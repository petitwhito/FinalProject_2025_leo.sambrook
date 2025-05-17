# KuaiRec Recommender System Project

This repository contains a complete recommender system implementation using the KuaiRec dataset. The project implements various recommender system approaches including collaborative filtering, content-based, hybrid, and sequence-aware recommendation methods.

## Project Structure

```
├── README.md
├── data
│   ├── KuaiRec
│   │   ├── data (Download data and put all csv here)
├── notebooks
│   ├── 1_data_preprocessing.ipynb
│   ├── 2_feature_engineering.ipynb
│   ├── 3_model_development.ipynb
│   ├── 4_recommendation_algorithm.ipynb
│   └── 5_Evaluation.ipynb
├── requirements.txt
└── src
    ├── data_processing.py
    ├── evaluation.py
    ├── feature_engineering.py
    ├── recommender.py
    ├── models
    │   ├── collaborative.py
    │   ├── content_based.py
    │   ├── hybrid.py
    │   └── sequence_aware.py
```

## Informations

Detailed explanations and implementation details can be found in the respective notebooks.

## Dataset

The KuaiRec dataset is available at:
https://drive.google.com/file/d/1qe5hOSBxzIuxBb1G_Ih5X-O65QElollE/view

**IMPORTANT:** After downloading the dataset, you must extract all CSV files and place them in the `data/KuaiRec/data` directory. The notebooks and source code are configured to look for the data files in this specific location, and the project will not run correctly without this setup.

The dataset contains user-video interaction data from Kuaishou, a popular short-video platform. It includes user features, video features, and user-video interactions with engagement metrics.

## Setup

To set up the project environment, run:

```
pip install -r requirements.txt
```

**WARNING:** This project requires significant memory resources. A minimum of 8GB RAM is recommended, with no other memory-intensive processes running simultaneously. Some operations, particularly during model training and evaluation, may consume substantial memory.

## Project Overview

This project follows a comprehensive pipeline for developing recommender systems:

1. **Data Processing**: Reading and preprocessing the KuaiRec dataset, computing statistics, and organizing data for model training.

2. **Feature Engineering**: Extracting and transforming features from user data, video data, and interaction data.

3. **Recommendation Models**:
   - Collaborative Filtering: User-based and item-based approaches
   - Content-Based: Recommending items based on content features
   - Hybrid: Combining collaborative and content-based approaches
   - Sequence-Aware: Considering the sequence of user interactions

4. **Evaluation**: Measuring the effectiveness of recommendation models using various metrics (RMSE, MAE, Precision, Recall, F1-Score).

## Main Results

### Performance Metrics at k=10

| Model          | Precision@10 | Recall@10 | MAP@10 | NDCG@10 |
|----------------|-------------|-----------|--------|---------|
| Collaborative  | 0.97        | 0.015     | 0.92   | 0.96    |
| Content-based  | 0.89        | 0.0135    | 0.85   | 0.90    |
| Sequence-aware | 0.95        | 0.0145    | 0.92   | 0.95    |
| Hybrid         | 0.95        | 0.0145    | 0.92   | 0.95    |
| Combined       | 0.96        | 0.0145    | 0.92   | 0.95    |

**Best Model**: The collaborative filtering approach consistently outperformed other models across all evaluation metrics, particularly in precision and recall. The combined model (ensemble approach) came in as a close second, showing the benefit of integrating multiple recommendation strategies.

## Author

Email: leo.sambrook@epita.fr
