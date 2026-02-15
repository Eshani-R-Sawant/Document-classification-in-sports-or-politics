# 📰 BBC News Classifier: Sport vs. Politics
### A Machine Learning & NLP Pipeline

This project is a comprehensive implementation of a text classification system designed to categorize BBC news articles into **Sport** or **Politics**. To demonstrate a deep understanding of the mathematical foundations of AI, the entire pipeline—from **feature extraction** to **machine learning models**—was built manually without standard libraries..

---

## 🎯 Project Goals
* **Feature Engineering:** Transform raw text into mathematical vectors using **Bag-of-Words**, **TF-IDF**, and **N-grams**.
* **Algorithmic Comparison:** Evaluate three distinct classification philosophies:
    1.  **Probabilistic:** Multinomial Naive Bayes.
    2.  **Discriminative:** Logistic Regression.
    3.  **Geometric:** Linear SVM.

---

## 🛠️ Feature Extraction Methodology

To convert raw text into a format understandable by Machine Learning models, we implemented a custom pipeline that ensures the models capture both word importance and local context.



#### **1. Bag of Words (BoW)**
The foundation of our feature vector. It represents text as a numerical "count" vector where each dimension corresponds to a unique word from the global vocabulary.

#### **2. N-Gram Model (Unigrams + Bigrams)**
To capture semantic context, we implemented **N-Grams**. 
* **Unigrams (n=1):** Individual words like "Minister".
* **Bigrams (n=2):** Pairs of consecutive words like "Prime Minister".

* **Why it matters:** This significantly increases accuracy by identifying phrases that carry category-specific meaning which is lost when words are split.

#### **3. TF-IDF (Term Frequency-Inverse Document Frequency)**
To prevent common words (like "the" or "said") from dominating the model, we apply **TF-IDF weighting**.



The final weight for any feature is calculated using:
* **TF:** Frequency of term $t$ in document $d$.
* **IDF:** $\log(\frac{N}{1 + df(t)})$, where $N$ is the total number of documents.
* **Final Weight:** $TF \times IDF$.

---

## 📊 Model Comparison
We implemented and compared three distinct mathematical frameworks:

| Model | Framework | Key Characteristic |
| :--- | :--- | :--- |
| **Naive Bayes** | Probabilistic | Uses Gaussian likelihoods and Laplace smoothing. |
| **Logistic Regression** | Discriminative | Optimized using manual Gradient Descent. |
| **Linear SVM** | Geometric | Maximizes decision margin via Hinge Loss. |



---

## 🚀 How to Run

### 1. Prerequisites
Ensure you have Python installed along with the following libraries:
```bash
### 2. Dataset Setup
Place the bbc_news.csv file in the same directory as the script. 

3. Execution
Run the main script using the following command:
Bash
python RollNumber_prob4.py

4. Output
Terminal: Displays progress of feature extraction and training.
<img width="695" height="451" alt="image" src="https://github.com/user-attachments/assets/8c7a13c2-e2d1-485b-a1c0-482e8ba4f79c" />



