# Flipkart Review Analysis
This script implements a text classification model that processes product reviews to predict whether a review is positive or negative, using machine learning techniques. Below is a detailed write-up explaining the steps, packages used, and the classifiers involved.

Purpose:
The model predicts whether a product review on Flipkart is positive or negative based on the review's text. It uses a binary classification approach where the label 1 represents positive reviews (rating ≥ 5), and 0 represents negative reviews (rating < 5).

Steps Followed:
Data Loading and Exploration:

The script begins by loading the flipkart_data.csv dataset using pandas. The first few rows of the data are displayed using data.head(), showing product reviews and their associated ratings.
Rating Analysis:

A count plot is generated to visualize the distribution of ratings using the seaborn package. This helps identify if the ratings are balanced or skewed.
Label Creation:

A new column, label, is added to the dataset. Reviews with a rating of 5 or more are labeled as 1 (positive), and reviews with a rating less than 5 are labeled as 0 (negative).
Text Preprocessing:

The nltk package is used to preprocess the reviews. The text is cleaned by:
Removing punctuation using regular expressions (re.sub).
Converting all words to lowercase.
Tokenizing the text and removing stopwords using the stopwords corpus from nltk.
The preprocess_text function applies these transformations to each review and stores the cleaned text in the review column.
Word Cloud Generation (for Positive Reviews):

A word cloud is created using the WordCloud class from the wordcloud package. It visualizes the most frequent words in positive reviews (label 1), giving insights into the common terms associated with positive feedback.
Text Vectorization:

The text data is vectorized using the TfidfVectorizer from sklearn. This converts the preprocessed text into numerical form by calculating the Term Frequency-Inverse Document Frequency (TF-IDF) for each word, retaining the top 2500 most important features.
Data Splitting:

The dataset is split into training and test sets using train_test_split from sklearn. 33% of the data is reserved for testing, while the remaining 67% is used for training. The splitting is stratified to maintain the same class distribution in both sets.
Model Training:

A DecisionTreeClassifier from sklearn is used as the classification model. The model is trained on the training data (X_train, y_train).
Model Evaluation:

The model is tested on the training set, and the accuracy score is computed using accuracy_score from sklearn.
A confusion matrix is generated to assess the performance of the model. The confusion matrix is displayed using ConfusionMatrixDisplay to show the true positive, true negative, false positive, and false negative values.
Packages Used:
pandas: Used for data manipulation and handling CSV files.
re: Regular expressions for text cleaning, such as removing punctuation.
seaborn: Used for data visualization (e.g., count plots).
sklearn: A key library for machine learning, used here for:
TfidfVectorizer: Text vectorization.
train_test_split: Splitting data into training and testing sets.
DecisionTreeClassifier: Decision tree algorithm for classification.
accuracy_score, confusion_matrix: Model evaluation metrics.
nltk: Natural Language Toolkit, used for text preprocessing, tokenization, and removing stopwords.
wordcloud: Used to generate a word cloud based on the most frequent words in positive reviews.
matplotlib: For plotting the word cloud and confusion matrix.
Classifiers Used:
Decision Tree Classifier: A supervised learning algorithm used to build a model that predicts a target label based on feature inputs. In this case, it classifies reviews as positive or negative based on the preprocessed review text. Decision trees are interpretable and handle both numerical and categorical data well.
