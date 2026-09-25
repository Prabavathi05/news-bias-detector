# 📰 News Article Bias Detector

A machine learning web app that predicts whether a news article leans **Left**, **Center**, or **Right** politically, and uses Google's Gemini API to generate a grounded, evidence-based explanation citing specific phrases from the article.

🔗 **Live demo:** [news-bias-detector-system.streamlit.app](https://news-bias-detector-system.streamlit.app/)
📂 **Source code:** [github.com/Prabavathi05/news-bias-detector](https://github.com/Prabavathi05/news-bias-detector)

---

## Overview

Most bias-detection projects stop at "here's a label." This project goes a step further: after predicting an article's political leaning using a classical NLP pipeline, it prompts Gemini to explain **why**, grounding the explanation in the article's own language rather than generic claims — similar in spirit to retrieval-augmented, evidence-based generation.

## Key Finding: TF-IDF beat Sentence-BERT embeddings

The most interesting result from this project wasn't which classifier won — it's that **lexical features (TF-IDF) consistently outperformed semantic embeddings (Sentence-BERT)** for this task.

| Feature Set | Model | Macro F1 | Accuracy |
|---|---|---|---|
| **TF-IDF** | **Logistic Regression** | **0.44** | 0.46 |
| TF-IDF | XGBoost | 0.43 | 0.52 |
| TF-IDF | SVM | 0.43 | 0.47 |
| TF-IDF | Naive Bayes | 0.29 | 0.48 |
| Sentence-BERT (MiniLM) | Logistic Regression | 0.39 | 0.40 |

**Why this happens:** political bias detection is largely a *lexical* task — it depends on specific word choices ("illegal" vs. "undocumented," "regime" vs. "government") and framing, not general topic meaning. Sentence embeddings compress text into a semantic-similarity space that can blur exactly the loaded word choices that signal bias, especially when two articles cover the same topic with different framing. Embedding truncation (MiniLM's 256-token limit vs. a median article length of ~509 words) likely compounded this, since TF-IDF sees the full document while embeddings saw less than half of most articles.

**Evaluation note:** Naive Bayes had the *highest accuracy (0.48)* of all TF-IDF models but the *worst macro-F1 (0.29)* — it achieved this by predicting "left" (the majority class) 93% of the time and almost never predicting "center" (2% recall). This is a concrete example of why accuracy alone is a misleading metric on imbalanced classes, and why macro-F1 was used as the primary metric throughout this project.

## Architecture

```
User pastes article text
   → TF-IDF vectorization (unigrams + bigrams, 20K features)
   → Logistic Regression classifier → predicted label + confidence
   → Gemini API (grounded prompt using the article's own text)
   → Streamlit UI displays prediction, confidence chart, and explanation
```

## Tech Stack

- **Data:** [AllSides-labeled news dataset](https://zenodo.org/records/7682915) (Qbias) — 21,747 articles labeled Left/Center/Right by AllSides editors
- **NLP / ML:** scikit-learn (TF-IDF, Logistic Regression, SVM, Naive Bayes), XGBoost, Sentence-Transformers (`all-MiniLM-L6-v2`)
- **GenAI:** Google Gemini API (`gemini-3.6-flash`) for grounded explanation generation
- **Deployment:** Streamlit Community Cloud
- **Other:** joblib (model persistence), pandas, GitHub

## Dataset

- 21,747 articles from AllSides, with class distribution: Left 47%, Right 33%, Center 20%
- Cleaned to remove nulls and articles under 100 characters → 21,108 usable articles
- Stratified 80/20 train/test split to preserve class balance in evaluation

## Model Selection Process

Four classifiers were benchmarked on the same TF-IDF features (Logistic Regression, Linear SVM, Multinomial Naive Bayes, XGBoost) to establish a baseline, then Sentence-BERT embeddings were tested as an alternative feature representation to see whether semantic features would outperform lexical ones. **Logistic Regression on TF-IDF was selected as the production model** — it had the best macro-F1 score, is fast, and is naturally interpretable (feature weights can be inspected directly, unlike a black-box embedding model).

## Explanation Layer (Gemini)

For each prediction, the article text and predicted label are passed to Gemini with an instruction to cite specific phrases from the article as evidence — rather than making generic claims about "biased language." This grounding approach reduces hallucinated reasoning and ties the explanation directly to verifiable text.

**Example output** (predicted: Right):
> "This article might be classified as right-leaning because it explicitly highlights that the infected individual was 'fully vaccinated'... Additionally, by underscoring that the patient experienced only 'mild symptoms that are improving'... the text frames the new variant as low-risk, aligning with right-leaning arguments against reinstated lockdowns..."

### Fault tolerance
The Gemini free tier has a strict rate limit (5 requests/minute) and occasionally returns `503 UNAVAILABLE` during high-demand periods. The app handles both cases with retry logic (exponential backoff) and **graceful degradation** — if the explanation call fails after retries, the app still displays the prediction and confidence, with a clear "temporarily unavailable" message instead of crashing. This was verified during an actual Gemini outage while building this project.

## Limitations & Honest Notes

- **Macro-F1 of ~0.44 is a genuinely hard ceiling for this task**, consistent with published academic work on article-level political bias classification. This should be read as a directional signal, not a ground-truth verdict.
- **Precision/recall on "Right" is ~0.44/0.43** — the model is correct on "right" predictions less than half the time. "Center" is the hardest class to detect (smallest class, least distinctive lexical signal).
- Labels reflect **AllSides' editorial judgment of the *source outlet/article*, not an objective or universally agreed-upon standard** of political bias.
- The classifier works on any text length (tested down to single headlines), but was trained on full articles — headline-only predictions should be treated as lower-confidence.

## Setup (local)

```bash
git clone https://github.com/Prabavathi05/news-bias-detector.git
cd news-bias-detector
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your_api_key_here"
```

Run:
```bash
streamlit run app.py
```

## Possible Future Improvements

- Expand training data with a non-US (e.g., Indian outlet-labeled) dataset for cross-market generalization
- Try chunk-and-average embeddings to test whether truncation, specifically, explains the embeddings underperformance
- Add SHAP-based feature importance to show *which words* drove each prediction, directly from the model rather than only via Gemini's narrative explanation
