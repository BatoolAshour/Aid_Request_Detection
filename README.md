# Aid Request Detection in Disaster Response Messages

Binary text classification of disaster messages using **TF-IDF** and **Sentence-BERT**, each paired with Logistic Regression.

**Course:** ARTI452 – Natural Language Processing (2026–2027, first semester) · **Section:** 7FA2 · **Group:** 7 · **Instructor:** Dr. Hayat Alfaghum

| Member | Responsibilities |
|--------|------------------|
| Batool Ashour | Data cleaning, exploratory analysis |
| Farah Almujaljel | Method 1 (TF-IDF) training and testing |
| Aryam Hassan | Method 2 (Sentence-BERT) training and testing |

---

## Overview

During a disaster, response teams receive thousands of messages from direct texts, news and social media. Only some of them are actual requests for help. This project builds classifiers that flag those messages automatically:

| Input | Output |
|-------|--------|
| A message sent during a disaster | `1` = Request for Aid, `0` = No Request |

**Experimental question:** *Which NLP method performs better at detecting aid requests in disaster response messages?*

Both methods use the same classifier (Logistic Regression), the same data split and the same tuning procedure, so any difference in results comes from the **text representation**.

## Dataset

[Disaster Response Messages (Figure Eight)](https://www.kaggle.com/datasets/sidharth178/disaster-response-messages) from Kaggle: English disaster-related messages labeled with 36 categories. We use the `request` column as the target.

| Step | Messages |
|------|---------:|
| Raw (messages merged with categories) | 26,216 |
| After removing empty, conflicting-label and duplicate messages | 26,157 |
| Of which: Request for Aid | 4,458 (17%) |

Key EDA findings:
- **Class imbalance:** about 83% No Request vs 17% Request for Aid.
- **Source matters:** 34% of direct messages are requests, compared with 5% of news and 7% of social media messages.
- Aid requests are slightly shorter than other messages on average.

## Methodology

1. **Cleaning:** remove empty, duplicate and conflicting-label messages. Two text versions are kept:
   - `text_clean` for TF-IDF: lowercased, letters only, URLs removed.
   - `text_light` for Sentence-BERT: only HTML entities, URLs and extra whitespace removed, so the sentence meaning stays intact.
2. **Split:** stratified 80 / 10 / 10 train / development / test split (20,925 / 2,616 / 2,616).
3. **Method 1, TF-IDF + Logistic Regression:** unigrams and bigrams (`min_df=2`, `max_df=0.95`, sublinear TF), giving 64,404 features. The vectorizer is fit on the training set only.
4. **Method 2, Sentence-BERT + Logistic Regression:** `all-MiniLM-L6-v2` encodes each message into a 384-dimensional embedding.
5. **Tuning (both methods):** `class_weight="balanced"` for the imbalance. `C` and the decision threshold are tuned on the development set to maximize Macro F1.
6. **Evaluation:** a single run on the shared test set using Accuracy, Macro F1 (main metric), per-class precision and recall, and confusion matrices.

| | TF-IDF | Sentence-BERT |
|---|---|---|
| Best `C` | 10 | 1 |
| Decision threshold | 0.67 | 0.73 |
| Dev Macro F1 (tuned) | 0.8194 | 0.8205 |

## Results (test set, 2,616 messages)

| Method | Accuracy | Macro F1 | Request Precision | Request Recall | False Positives | False Negatives |
|--------|---------:|---------:|------------------:|---------------:|----------------:|----------------:|
| TF-IDF + LR | **0.9128** | **0.8358** | **0.7884** | 0.6682 | **80** | 148 |
| Sentence-BERT + LR | 0.9083 | 0.8336 | 0.7464 | **0.6996** | 106 | **134** |

- The two methods perform **almost the same**. TF-IDF is ahead by only 0.002 in Macro F1, and Sentence-BERT scored slightly higher on the development set, so the gap is within normal variation.
- **TF-IDF is more precise** (fewer false alarms). **Sentence-BERT has higher recall**, so it misses fewer real aid requests, which matters most in disaster response.
- The models disagree on 188 test messages, which suggests they make different kinds of mistakes.
- TF-IDF depends heavily on keywords such as *help*, *need*, *tent*, *hungry*, *food* and *water*. Because of this, both models wrongly flag sentences like *"Volunteers distributed food and water to affected families"* as requests.

## Repository Structure

```
Aid_Request_Detection/
├── Aid_Request_Detection.ipynb   # Full pipeline: EDA → cleaning → both methods → evaluation → demo
└── README.md
```

When run, the notebook also creates `tfidf_logreg_artifact.joblib`, `best_sbert_logistic_model.pkl`, `best_sbert_threshold.npy`, `model_comparison_results.csv` and the dataset split CSVs.

## How to Run

The notebook was developed in **Google Colab** (a GPU runtime is recommended for Sentence-BERT encoding).

1. Open `Aid_Request_Detection.ipynb` in Colab.
2. Install the dependencies if needed:
   ```bash
   pip install kagglehub sentence-transformers scikit-learn pandas numpy matplotlib seaborn joblib
   ```
3. Run all cells. The dataset downloads automatically with `kagglehub`.
4. Use `predict_message("your message")` in Section 11 to classify new messages with both models.

> The `google.colab.files.download(...)` cells only work inside Colab. Skip them if you run the notebook locally.

## References

1. Figure Eight, "Disaster Response Messages," Kaggle. https://www.kaggle.com/datasets/sidharth178/disaster-response-messages
2. N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," in *Proc. EMNLP-IJCNLP*, 2019.
3. Scikit-learn documentation: [`TfidfVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html), [`LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), [`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html).
