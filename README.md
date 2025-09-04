# 📝 Sentiment Analysis with Machine Learning & NLP

Sentiment Analysis is one of the most widely studied applications of Natural Language Processing (NLP). It enables us to automatically identify opinions expressed in text and classify them as **positive, negative, or neutral**.  

In this project, I built a **complete sentiment analysis pipeline** that not only preprocesses raw text but also explores **three different NLP feature extraction techniques** and evaluates **six machine learning algorithms** with **hyperparameter tuning** to find the most effective approach.

---

## 🎯 Project Objectives
1. **Understand text sentiment trends** by applying multiple feature engineering techniques.  
2. **Compare performance of ML models** across feature extraction methods.  
3. **Identify the most effective model + vectorization combination** for sentiment classification.  
4. Provide a **reproducible end-to-end pipeline** that can be extended to real-world datasets.

---

## 📌 Approach

### 1. Data Preprocessing
Raw text often contains noise, inconsistencies, and irrelevant tokens. The following preprocessing steps were applied:
- Converting text to lowercase
- Removing punctuation, numbers, and special characters
- Tokenization
- Stopword removal
- Lemmatization to reduce words to their base form  

This ensured cleaner input for downstream NLP tasks.

---

### 2. Feature Engineering
To convert textual data into machine-readable vectors, three popular approaches were used:

- **Bag of Words (BoW):** Represents text as simple frequency counts.  
- **TF-IDF:** Weights words based on their importance across documents.  
- **Word2Vec:** Creates dense vector embeddings that capture semantic meaning of words.  

This allowed us to capture both **syntactic structure** (BoW, TF-IDF) and **semantic relationships** (Word2Vec).

---

### 3. Machine Learning Models
For each vectorization method, the following classifiers were trained:
- Logistic Regression
- Random Forest
- Decision Tree
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Support Vector Machine (SVM)

---

### 4. Hyperparameter Tuning
Every model was fine-tuned using **GridSearchCV** and **RandomizedSearchCV** to optimize parameters such as:
- Logistic Regression: `C`, regularization strength  
- Random Forest: `n_estimators`, `max_depth`  
- Decision Tree: `criterion`, `max_depth`  
- KNN: `n_neighbors`, `weights`  
- Naive Bayes: smoothing parameters  
- SVM: `C`, `kernel`, `gamma`  

This ensured a fair and optimized comparison across models.

---

### 5. Evaluation
Models were evaluated on:
- **Accuracy** (primary comparison metric)  
- **Precision, Recall, F1-score** (to balance class imbalances)  
- **Confusion Matrices** (for error analysis)  

Finally, results were visualized for **all classifiers across all vectorization techniques**.

---

## 📊 Results

### Accuracy Comparisons
- **Bag of Words (BoW)**  
  ![BoW Accuracy](images/Accuracy%20Comparison%20for%20BoW.png)

- **TF-IDF**  
  ![TF-IDF Accuracy](images/Accuracy%20Comparison%20for%20TF-IDF.png)

- **Word2Vec**  
  ![Word2Vec Accuracy](images/Accuracy%20Comparison%20for%20Word2Vec.png)


📌 ## **Key Findings:**
## **Model Evaluation Insights – Emotion Classification**
---

### **Bag-of-Words (BoW)**
- **Overall Performance:** Most models perform at the same level with **Accuracy = 0.6** and **F1 = 0.45**, except **Decision Tree** which drops to **Accuracy = 0.2, F1 = 0.24**.  
- **Best Models:** Logistic Regression, Naive Bayes, Random Forest, SVM, and KNN all tie with **F1 = 0.45**.  
- **Observations:**  
  - Logistic Regression, Naive Bayes, Random Forest, SVM, KNN → **Precision = 0.36, Recall = 0.6** → models capture positives well but sacrifice precision.  
  - Decision Tree → **Precision = 0.30, Recall = 0.2** → poor recall limits usefulness.  

---

### **TF-IDF**
- **Overall Performance:** Very similar to BoW → most models **Accuracy = 0.6, F1 = 0.45**, with **Decision Tree lagging (F1 = 0.24)**.  
- **Best Models:** Logistic Regression, Naive Bayes, Random Forest, SVM, KNN (all tied with F1 = 0.45).  
- **Observations:**  
  - Decision Tree → still weak with **low recall (0.2)**.  
  - Others → **Precision = 0.36, Recall = 0.6**, consistent across classifiers.  
- **Conclusion:** TF-IDF does **not add a clear performance advantage** over BoW in this dataset.  

---

### **Word2Vec**
- **Overall Performance:** Most models stabilize at **Accuracy = 0.6, F1 = 0.45**, with **Decision Tree slightly better than before (F1 = 0.40)**.  
- **Best Models:** Logistic Regression, Random Forest, SVM, and KNN all tie at **F1 = 0.45**.  
- **Observations:**  
  - Decision Tree → **Precision = 0.40, Recall = 0.40**, balanced but still weaker overall.  
  - Logistic Regression, Random Forest, SVM, KNN → **Precision = 0.36, Recall = 0.6**, same moderate pattern seen in BoW and TF-IDF.  

---

### **Key Takeaways**
- Across **BoW, TF-IDF, and Word2Vec**, most classifiers (except Decision Tree) show **similar moderate performance** with **F1 ≈ 0.45**.  
- **Decision Tree consistently underperforms**, especially in recall.  
- **Hyperparameter tuning equalized results** across models — before tuning, KNN stood out slightly, but after tuning, most models converged to similar scores.  
- All three feature representations **(BoW, TF-IDF, Word2Vec)** give almost the same results, with no clear winner.  

---

### **Recommendation**

Based on the evaluation results, **Logistic Regression with TF-IDF features** is the best choice:

- It provides **stable performance** (Accuracy = 0.6, F1 = 0.45), comparable to the best-performing models.  
- TF-IDF representation captures informative words better than BoW, making it more reliable for text data.  
- Logistic Regression offers **interpretability**, allowing us to clearly understand which words drive classification decisions.  

If robustness against noisy data is preferred (rather than interpretability), **Random Forest with Word2Vec** can be considered as an alternative.  

#### **Final Recommendation:**

**Logistic Regression + TF-IDF** (primary choice), with **Random Forest + Word2Vec** as a secondary option.



---

## 🛠 Tech Stack
- **Language**: Python 3.x  
- **Libraries**:  
  - Data Processing → `pandas`, `numpy`  
  - NLP → `nltk`, `spacy`, `gensim`  
  - Machine Learning → `scikit-learn`  
  - Visualization → `matplotlib`, `seaborn`  

---


