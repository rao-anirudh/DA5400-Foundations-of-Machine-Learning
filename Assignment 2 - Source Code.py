# Name: Anirudh Rao
# Roll No: BE21B004

# Importing necessary libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
import string
import warnings
import random

warnings.filterwarnings('ignore')

# Reading the emails

cwd = os.getcwd()
spam_emails = os.listdir(cwd + '\\enron1\\spam')
ham_emails = os.listdir(cwd + '\\enron1\\ham')

# Creating a train-test split

print("\nStarting train-test split")

random.seed(5400)

train_spam_emails = random.sample(spam_emails, int(0.8 * len(spam_emails)))
train_ham_emails = random.sample(ham_emails, int(0.8 * len(ham_emails)))
test_spam_emails = [email for email in spam_emails if email not in train_spam_emails]
test_ham_emails = [email for email in ham_emails if email not in train_ham_emails]

os.makedirs(cwd + '\\train\\spam')
os.makedirs(cwd + '\\train\\ham')
os.makedirs(cwd + '\\test\\spam')
os.makedirs(cwd + '\\test\\ham')

for email in train_spam_emails:
    shutil.copy(cwd + '\\enron1\\spam\\' + email, cwd + '\\train\\spam')

for email in train_ham_emails:
    shutil.copy(cwd + '\\enron1\\ham\\' + email, cwd + '\\train\\ham')

for email in test_spam_emails:
    shutil.copy(cwd + '\\enron1\\spam\\' + email, cwd + '\\test\\spam')

for email in test_ham_emails:
    shutil.copy(cwd + '\\enron1\\ham\\' + email, cwd + '\\test\\ham')

# Preprocessing

# Defining lemmatization function

print("\nStarting preprocessing")

import spacy

lemmatizer = spacy.load("en_core_web_sm")

# Defining stop words

import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

vocabulary = {"IS_NUMERIC": np.inf}
spam = list()
ham = list()

for email in os.listdir(cwd + '\\train\\spam'):
    with open(cwd + '\\train\\spam\\' + email, 'r', errors='ignore') as f:
        content = f.read()
        f.close()
        processed_content = set()
        content = content.lower().replace("\n", " ").translate(str.maketrans('', '', string.punctuation)).replace("  ",
                                                                                                                  " ")
        content = " ".join([token.lemma_ for token in lemmatizer(content)]).lower()
        words = content.split()
        words = [word for word in words if word not in stop_words]
        for word in words:
            if word.isalpha() and word not in vocabulary:
                vocabulary[word] = 1
            elif word.isalpha():
                vocabulary[word] += 1
        processed_content.update(words)
        processed_content = {x if x.isalpha() else "IS_NUMERIC" for x in processed_content}
        spam.append(list(processed_content))

for email in os.listdir(cwd + '\\train\\ham'):
    with open(cwd + '\\train\\ham\\' + email, 'r', errors='ignore') as f:
        content = f.read()
        f.close()
        processed_content = set()
        content = content.lower().replace("\n", " ").translate(str.maketrans('', '', string.punctuation)).replace("  ",
                                                                                                                  " ")
        content = " ".join([token.lemma_ for token in lemmatizer(content)]).lower()
        words = content.split()
        words = [word for word in words if word not in stop_words]
        for word in words:
            if word.isalpha() and word not in vocabulary:
                vocabulary[word] = 1
            elif word.isalpha():
                vocabulary[word] += 1
        processed_content.update(words)
        processed_content = {x if x.isalpha() else "IS_NUMERIC" for x in processed_content}
        ham.append(list(processed_content))

# Low-frequency pruning

threshold = (0.1 / 100) * len(ham + spam)
pruned_vocabulary = {word: count for word, count in vocabulary.items() if count >= threshold}

# Feature engineering (binary feature vector creation)

print("\nStarting feature engineering")

spam_df = pd.DataFrame(columns=list(pruned_vocabulary.keys()), data=np.zeros((len(spam), len(pruned_vocabulary))))
i = 0
for email in spam:
    for word in email:
        if word in pruned_vocabulary.keys():
            spam_df.loc[i, word] = 1
    i += 1
spam_df["EMAIL_TYPE"] = 1

ham_df = pd.DataFrame(columns=list(pruned_vocabulary.keys()), data=np.zeros((len(ham), len(pruned_vocabulary))))
i = 0
for email in ham:
    for word in email:
        if word in pruned_vocabulary.keys():
            ham_df.loc[i, word] = 1
    i += 1
ham_df["EMAIL_TYPE"] = 0

train_df = pd.concat([spam_df, ham_df])


# Train-validation split

from sklearn.model_selection import train_test_split

X = train_df.drop("EMAIL_TYPE", axis=1)
y = train_df["EMAIL_TYPE"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=5400)


# Defining the Naive Bayes classifier

class NaiveBayesClassifier:

    def __init__(self, alpha=1):

        self.p = None
        self.class_0_probabilities = None
        self.class_1_probabilities = None
        self.vocabulary = None
        self.class_0 = None
        self.class_1 = None
        self.alpha = alpha
        self.is_fitted = False

    def fit(self, X, y):

        self.class_0, self.class_1 = np.unique(y)[0], np.unique(y)[1]

        self.p = len(y[y == self.class_1]) / len(y)

        class_0_data = X[y == self.class_0]
        self.class_0_probabilities = {word: sum(class_0_data[word]) + self.alpha for word in class_0_data.columns}
        self.class_0_probabilities = {
            word: (self.class_0_probabilities[word] / sum(self.class_0_probabilities.values())) for word in
            self.class_0_probabilities}

        class_1_data = X[y == self.class_1]
        self.class_1_probabilities = {word: sum(class_1_data[word]) + self.alpha for word in class_1_data.columns}
        self.class_1_probabilities = {
            word: (self.class_1_probabilities[word] / sum(self.class_1_probabilities.values())) for word in
            self.class_1_probabilities}

        self.vocabulary = list(self.class_0_probabilities.keys())

        self.is_fitted = True

    def predict(self, X):

        if not self.is_fitted:
            raise Exception("Model not fitted")

        words_present = [word for (word, present) in zip(self.vocabulary, X) if present]

        class_0_prob = np.prod([self.class_0_probabilities[word] for word in words_present]) * self.p
        class_1_prob = np.prod([self.class_1_probabilities[word] for word in words_present]) * (1 - self.p)

        if class_0_prob >= class_1_prob:
            return self.class_0
        else:
            return self.class_1

    def predict_from_path(self, path):

        if not self.is_fitted:
            raise Exception("Model not fitted")

        test_emails = list()

        for email in os.listdir(path):
            with open(path + email, 'r', errors='ignore') as f:
                content = f.read()
                f.close()
                processed_content = set()
                content = content.lower().replace("\n", " ").translate(
                    str.maketrans('', '', string.punctuation)).replace("  ", " ")
                content = " ".join([token.lemma_ for token in lemmatizer(content)]).lower()
                words = content.split()
                processed_content.update(words)
                processed_content = {x if x.isalpha() else "IS_NUMERIC" for x in processed_content}
                test_emails.append(list(processed_content))

        test_df = pd.DataFrame(columns=self.vocabulary, data=np.zeros((len(test_emails), len(self.vocabulary))))
        i = 0
        for email in test_emails:
            for word in email:
                if word in self.vocabulary:
                    test_df.loc[i, word] = 1
            i += 1

        y_pred = test_df.apply(lambda x: self.predict(x), axis=1)

        return y_pred


# Hyperparameter tuning

from sklearn.metrics import accuracy_score

print("\nStarting classifier training and hyperparameter tuning\n")
scores = dict()
for alpha in np.linspace(0, 1, 11):
    classifier = NaiveBayesClassifier(alpha=alpha)
    classifier.fit(X_train, y_train)
    y_val_pred = X_val.apply(lambda x: classifier.predict(x), axis=1)
    scores[alpha] = accuracy_score(y_val, y_val_pred)

plt.figure(dpi=150)
plt.xlabel("Alpha")
plt.ylabel("Validation Accuracy")
plt.plot(list(scores.keys()), list(scores.values()), marker='o')
plt.show()

print(pd.DataFrame(scores.items(), columns=["Alpha", "Validation Accuracy"]))

best_alpha = list(scores.keys())[np.argmax(list(scores.values()))]
best_classifier = NaiveBayesClassifier(alpha=best_alpha)
best_classifier.fit(X, y)

print(f"\nBest alpha: {best_alpha}\n")

# Test performance

print(f"Final testing")

spam_test_pred = best_classifier.predict_from_path(cwd + '\\test\\spam\\')
ham_test_pred = best_classifier.predict_from_path(cwd + '\\test\\ham\\')

true_spam = len(spam_test_pred[spam_test_pred == 1])
true_ham = len(ham_test_pred[ham_test_pred == 0])

false_spam = len(spam_test_pred[spam_test_pred == 0])
false_ham = len(ham_test_pred[ham_test_pred == 1])

test_accuracy = (true_spam + true_ham) / (true_spam + true_ham + false_spam + false_ham)
print(f"\nTest Accuracy: {test_accuracy}")
