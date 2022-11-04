
# Text Classification (The Gutenbergs' books)

Text classification is one of the major tasks of AI in general, NLP in particular. Having five Gutenberg books, this report discusses the methodologies and models with different transformation techniques that have been applied to reach the best accuracy that the champion model achieves by correctly classifying unseen text to the corresponding book.

## Introduction
Text can be a rich source of information, however extracting insights from it is not an easy task due to its unstructured nature. The overall objective is to produce classification predictions and compare them; analyze the pros and cons of algorithms and generate and communicate the insights so, the implementation is needed to check the accuracy of each model going to be used and select the champion model. the best language to be used in such problems is python with its vast libraries.

## Dataset
The Gutenberg dataset represents a corpus of over 60,000 book texts, their authors and titles. The data has been scraped from the Project Gutenberg website using a custom script to parse all bookshelves. we have taken five different samples of Gutenberg digital books that are of five different authors, that we think are of the criminal same genre and are semantically the same. The books are Criminal Sociology by Enrico Ferri. Crime: Its Cause and Treatment by Clarence Darrow. The Pirates' Who's Who by Philip Gosse. Celebrated Crimes by Alexandre Dumas and Buccaneers and Pirates of Our Coasts by Frank Richard. 

## Installation 
import nltk
import pandas as pd
import re
import numpy as np
nltk.download("gutenberg")
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.neighbors import KNeighborsClassifier # knn classifier
import seaborn as sns
import wordcloud
import mlxtend.evaluate
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import Image  
import pydotplus
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings(action='ignore')
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2, f_classif
from mlxtend.evaluate import bias_variance_decomp
from scipy import sparse

#  Solution (G10_textClassification.ipynb)
### This contains a series of steps to manipulate the data ("Gutenberg's couple of digital books") and serialize them:
       1. Make Preprocessing and Data Cleaning
			- Work on 5 fictions books with different authors.
			- Have 1000 rows each row has 100 words.
			- Use label encoder "on Author".
			- Clean the data using regex.
			- Remove the stop words too.
       2. Use Feature Engineering ("Feature selection" and "Features Reduction")
			- Use "SelectPercentile" for the most 20% important features.
			- Use "BOW", "N-Gram", and "TFiDF" to create new features.
       3. Train 3 algorithms with 3 transformer
	   4. Evaluate each model "cv ,mean accuracy and std
	   5. Display bais and variance for All models to choose the champion model.
	   6. Select The champion model ("TFiDF With Support Vector Machine")
	   7. Apply Error Analysis on The champion model.
## Thank You
Ahmed Abdo Amin Abdo
