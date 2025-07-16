# 🛍️ Sentiment Analysis of Amazon Product Reviews

This project is a smart NLP-based sentiment analysis tool that classifies Amazon product reviews into **Positive**, **Negative**, or **Neutral** categories using a **Naive Bayes classifier** and classic text processing techniques.

Built by **Utkarsh Gupta** — optimized for fast, reliable, and interpretable results from real-world review data.

---

## Features

- Cleaned and preprocessed Amazon review text
- Applied **tokenization, stopword removal, stemming**, and **TF-IDF vectorization**
- Trained with **Multinomial Naive Bayes** for high efficiency and accuracy
- Supports **real-time sentiment prediction** via a Flask web app
- Easy to deploy or extend for other review platforms (e.g., Flipkart, Myntra)

---

## Tech Stack

| Component      | Tool Used                      |
|----------------|--------------------------------|
| Language        | Python                        |
| Libraries       | NLTK, Scikit-learn, Pandas     |
| Model           | Multinomial Naive Bayes        |
| Vectorization   | TF-IDF                         |
| Web Framework   | Flask                          |
| Deployment      | (Optional) Render / Streamlit  |

---

## Preprocessing Pipeline

1. Convert to lowercase and remove special characters
2. Tokenization (splitting text into words)
3. Stopword removal (e.g., "the", "is", "in")
4. Stemming using Porter Stemmer
5. TF-IDF Vectorization to extract features

---

## Model Training & Testing

- **Algorithm**: Multinomial Naive Bayes
- **Dataset**: Labeled Amazon product reviews
- **Train-Test Split**: 80-20
- **Accuracy**: ~85–90% on test data
- **Evaluation**: Confusion Matrix, Precision, Recall, F1-score

---

## How to Run Locally

```bash
git clone https://github.com/your-username/sentiment-analysis-of-amazon-product-reviews.git
cd sentiment-analysis-of-amazon-product-reviews
pip install -r requirements.txt
python app.py
```

Visit: `http://localhost:5000` to analyze reviews live!

---

## Sample Input

```
"This product was amazing and arrived on time!"
```

**Prediction**: Positive 

---

## Applications

- Product review monitoring
- E-commerce feedback analysis
- Brand reputation tracking
- Customer satisfaction prediction

---

## Limitations

- May struggle with sarcasm or complex sentiment
- Works best on English-language text
- Simpler than transformer-based models (like BERT)

---

## Author

**Utkarsh Gupta**  
B.Tech CSE (Data Science) | PSIT Kanpur  

