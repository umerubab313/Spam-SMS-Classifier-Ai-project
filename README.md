📩 Spam SMS Classification using Machine Learning
📌 Project Overview

This project focuses on building an end-to-end Spam SMS Classification system using classical Machine Learning and Natural Language Processing (NLP) techniques.

The goal is to automatically classify SMS messages as Spam or Ham (Not Spam) by performing deep data analysis, text preprocessing, feature extraction, and model comparison.

Rather than relying on a single model, multiple ML algorithms were trained and evaluated to understand their strengths and weaknesses on text-based data.

🧠 Key Concepts Covered

Exploratory Data Analysis (EDA)

Text preprocessing & cleaning

Feature extraction using NLP techniques

Training and evaluation of multiple ML models

Performance comparison and error analysis

🗂 Dataset

Type: SMS text messages

Labels: spam, ham

The dataset contains real-world SMS messages with class imbalance, making it suitable for evaluating precision and recall.

🔍 Exploratory Data Analysis (EDA)

Performed detailed analysis to understand:

Class distribution (Spam vs Ham)

Message length patterns

Word frequency in spam and ham messages

Common keywords used in spam messages

Visualizations were used to identify trends and biases in the data before modeling.

🧹 Text Preprocessing

The raw SMS data was cleaned and normalized using the following steps:

Lowercasing text

Removing punctuation and special characters

Tokenization

Stopword removal

Stemming / Lemmatization (where applicable)

This step ensured that the model learns from meaningful textual patterns rather than noise.

🧠 Feature Engineering

Text data was converted into numerical form using:

Bag of Words (BoW)

TF-IDF Vectorization

These techniques helped capture the importance of words while reducing the impact of frequently occurring but less informative terms.

🤖 Machine Learning Models Used

The following models were trained and evaluated:

Naive Bayes

Logistic Regression

Support Vector Machine (SVM)

Random Forest (if applicable)

Each model was compared using consistent evaluation metrics to ensure fair analysis.

📊 Model Evaluation

Models were evaluated using:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

Special attention was given to false positives and false negatives, as misclassifying spam can have real-world consequences.

🏆 Results & Insights

Simpler probabilistic models like Naive Bayes performed very well due to the nature of text data.

TF-IDF generally improved performance compared to raw Bag of Words.

Precision was prioritized to reduce false spam detection.

Model comparison helped identify trade-offs between accuracy and interpretability.

🛠 Technologies Used

Python

NumPy

Pandas

Matplotlib / Seaborn

scikit-learn

NLTK

📁 Project Structure
├── spam-detection.ipynb
├── dataset/
│   └── spam.csv
├── README.md

🚀 How to Run the Project

Clone the repository:

git clone <repo-link>


Install dependencies:

pip install -r requirements.txt


Open the notebook:

jupyter notebook spam-detection.ipynb

🎯 What I Learned

How to build a complete ML pipeline for text classification

Importance of data cleaning in NLP tasks

How different ML models behave on textual data

How to properly evaluate and compare models

How to analyze errors instead of relying only on accuracy

📌 Future Improvements

Use n-gram based feature extraction

Hyperparameter tuning

Deploy as a web application

Extend the model to email spam detection

🧑‍💻 Author

Ume Rubab
Bachelor’s in Computer Science
Interested in AI, Machine Learning, and Research
