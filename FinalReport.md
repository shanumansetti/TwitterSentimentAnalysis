### **Final Report: Sentiment Analysis for Tweet Classification**

---

#### **1. Introduction**

Twitter, a popular social media platform, serves as a vast reservoir for public opinions and emotions. Unfortunately, the presence of harmful or hateful content can negatively affect users' experiences. In response to this issue, we set out to create a **Natural Language Processing (NLP)**-based model that can automatically classify tweets based on their sentiment, with a particular focus on detecting negative content to prevent its widespread dissemination.

The dataset consists of tweets, each labeled with a sentiment classification: **Positive**, **Negative**, or **Neutral**. The task is to predict the sentiment of these tweets, providing a method to automatically filter harmful content from the platform.

---

#### **2. Dataset Overview**

The dataset comprises **27,481 rows** with the following key columns:

- **text**: Full content of the tweet.
- **sentiment**: Sentiment of the tweet (positive, negative, neutral).

We aim to develop and evaluate a machine learning model capable of accurately classifying tweets into these sentiment categories.

---

#### **3. Exploratory Data Analysis (EDA)**

- **Shape of the dataset**: 27,481 tweets with 2 columns: `text` and `sentiment`.
- **Sentiment distribution**: The dataset shows an imbalanced distribution of sentiments, with **Neutral** being the most frequent sentiment.

#### Key Insights:
- **Word Count**: A feature was added to analyze the average number of words in tweets, with some tweets having a word count up to several hundred.
- **Character Count**: We also looked at the number of characters in the tweets.
- **Stopwords**: We identified and analyzed the impact of stopwords in tweets.
- **Special Characters & Hashtags**: Tweets often contain special characters, hashtags, and mentions (e.g., `@user`).

---

#### **4. Data Preprocessing**

The preprocessing steps involved:
1. **Text Cleaning**: Conversion of text to lowercase and removal of special characters.
2. **Stopwords Removal**: Common words like “is”, “the”, etc., were removed using the NLTK stopwords list.
3. **Word Frequency Analysis**: Frequent words like "I'm" and "****" were removed as they didn't provide meaningful information.
4. **Stemming**: We applied stemming to reduce words to their root form (e.g., “running” to “run”).
5. **Rare Word Removal**: We removed words that occurred very infrequently in the dataset, as they likely introduced noise.

---

#### **5. Model Development**

Several machine learning models were tested, including:

- **Logistic Regression**
- **Naive Bayes Classifier**
- **Support Vector Classifier (SVC)**
- **Decision Tree Classifier**
- **Random Forest Classifier**

For each model, we used **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert text data into numerical features.

---

#### **6. Model Evaluation**

##### **Logistic Regression (LR) Results:**
- **Accuracy**: 82.1%
- **F1-Score**: 81.2%
- **AUC-ROC**: 87.5%

##### **Naive Bayes (NB) Results:**
- **Accuracy**: 78.5%
- **F1-Score**: 77.8%
- **AUC-ROC**: 84.3%

##### **Key Evaluation Metrics:**
- **Accuracy**: Measures the overall correctness of the model.
- **Precision**: The proportion of true positive results among all predicted positives.
- **Recall**: The proportion of actual positives that were correctly identified.
- **F1-Score**: The harmonic mean of precision and recall, which is crucial in imbalanced datasets.

Given the imbalance in the dataset, **F1-Score** became the primary metric for evaluation.

---

#### **7. Model Comparison**

We compared the performance of the models using a confusion matrix and key classification metrics. The **Logistic Regression** model outperformed other models, particularly in terms of **AUC-ROC** and **F1-Score**, making it the best choice for this task.

---

#### **8. Final Model: Logistic Regression**

- After evaluating the models, we selected **Logistic Regression** as the final model for sentiment classification. Its high **F1-Score** and **AUC-ROC** make it the most reliable model for identifying negative sentiment, which is of prime importance for this task.

---

#### **9. Conclusion**

- **Challenges**: The imbalanced nature of the sentiment labels posed a significant challenge. We focused on **F1-Score** rather than accuracy to ensure balanced performance across different classes.
- **Future Work**: Future improvements could involve applying **Deep Learning models** like **LSTM** (Long Short-Term Memory) networks or **BERT** (Bidirectional Encoder Representations from Transformers), which have been shown to outperform traditional machine learning models for text classification tasks.

In summary, this project demonstrates how sentiment analysis can be effectively used to detect harmful content on platforms like Twitter, contributing to the creation of a safer and healthier online environment.

---

#### **10. Recommendations for Deployment**
- **Automated Filtering**: The model can be integrated into Twitter’s content moderation system to automatically flag or filter harmful content.
- **Real-Time Analysis**: Given the real-time nature of Twitter, implementing this model in real-time could allow for immediate responses to harmful content.

---

#### **Appendix: Code Summary**

The full Python code for data preprocessing, feature extraction (TF-IDF), model building, and evaluation can be found in the notebook provided. Key libraries used include **scikit-learn**, **NLTK**, and **matplotlib** for visualizations. The models were tested and compared using evaluation metrics such as **accuracy**, **precision**, **recall**, **F1-Score**, and **AUC-ROC**.

--- 

This concludes the final report on sentiment analysis for tweet classification.