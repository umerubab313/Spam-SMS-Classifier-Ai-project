📩 #Spam SMS Classification using Machine Learning

<b>📌 Project Overview</b>

This project focuses on building an end-to-end Spam SMS Classification system using classical Machine Learning and Natural Language Processing (NLP) techniques.

The goal is to automatically classify SMS messages as Spam or Ham (Not Spam) by performing deep data analysis, text preprocessing, feature extraction, and model comparison.

Rather than relying on a single model, multiple ML algorithms were trained and evaluated to understand their strengths and weaknesses on text-based data.

<b>🧠 Key Concepts Covered</b>

Exploratory Data Analysis (EDA)

Text preprocessing & cleaning

Feature extraction using NLP techniques

Training and evaluation of multiple ML models

Performance comparison and error analysis

<b>🗂 Dataset</b>

Type: SMS text messages

Labels: spam, ham

The dataset contains real-world SMS messages with class imbalance, making it suitable for evaluating precision and recall.

<b>🔍 Exploratory Data Analysis (EDA)</b>

Performed detailed analysis to understand:

Class distribution (Spam vs Ham)

Message length patterns

Word frequency in spam and ham messages

Common keywords used in spam messages

Visualizations were used to identify trends and biases in the data before modeling.

<b>🧹 Text Preprocessing</b>

The raw SMS data was cleaned and normalized using the following steps:

Lowercasing text

Removing punctuation and special characters

Tokenization

Stopword removal

Stemming / Lemmatization (where applicable)

This step ensured that the model learns from meaningful textual patterns rather than noise.

<b>🧠 Feature Engineering</b>

Text data was converted into numerical form using:

Bag of Words (BoW)

TF-IDF Vectorization

These techniques helped capture the importance of words while reducing the impact of frequently occurring but less informative terms.

<b>🤖 Machine Learning Models Used</b>

The following models were trained and evaluated:

Naive Bayes

Logistic Regression

Support Vector Machine (SVM)

Random Forest (if applicable)

Each model was compared using consistent evaluation metrics to ensure fair analysis.

<b>📊 Model Evaluation</b>

Models were evaluated using:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

Special attention was given to false positives and false negatives, as misclassifying spam can have real-world consequences.

<b>🏆 Results & Insights</b>

Simpler probabilistic models like Naive Bayes performed very well due to the nature of text data.

TF-IDF generally improved performance compared to raw Bag of Words.

Precision was prioritized to reduce false spam detection.

Model comparison helped identify trade-offs between accuracy and interpretability.

<b>🛠 Technologies Used</b>

Python

NumPy

Pandas

Matplotlib / Seaborn

scikit-learn

NLTK

<b>Open the notebook:</b>

jupyter notebook spam-detection.ipynb

<b>🎯 What I Learned</b>

How to build a complete ML pipeline for text classification

Importance of data cleaning in NLP tasks

How different ML models behave on textual data

How to properly evaluate and compare models

How to analyze errors instead of relying only on accuracy

<b>📌 Future Improvements</b>

Use n-gram based feature extraction

Hyperparameter tuning

Deploy as a web application

Extend the model to email spam detection

<b>🧑‍💻 Author<b/>

Ume Rubab
Bachelor’s in Computer Science
Interested in AI, Machine Learning, and Research
