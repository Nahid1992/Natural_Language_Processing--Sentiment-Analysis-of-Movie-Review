import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
#from kaggleWord2VecUtility import kaggleWord2VecUtility
import pandas as pd
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
stopWords = set(stopwords.words('english'))

#nltk.download()


def load_data(foldername, filename_type):
	#Read Data
	type = pd.read_csv(os.path.join(os.path.dirname(__file__),foldername,filename_type),header=0,delimiter="\t",quoting=3)
	
	print('Data Loaded...')
	return type
	
	
def sentence2word(train):
	#for i in range(0, len(train["review"])):
	#	clean_train_reviews.append(" ".join(kaggleWord2VecUtility.review_to_wordlist(train["review"][i],True)))
	clean_wordList = []
	for i in range(0,len(train["review"])):
		review = BeautifulSoup(train["review"][i],"html5lib").get_text()
		review = re.sub("[^a-zA-Z]"," ", review)
		review = review.lower()
		wordList_filtered = []
		wordList = nltk.word_tokenize(review)
		for w in wordList:
			if w not in stopWords:
			    wordList_filtered.append(w)			
		clean_wordList.append(" ".join(wordList_filtered))
	return clean_wordList
	
	
def data_vectorize(clean_data):
	print('Creating the bag of words...')
	vectorizer = CountVectorizer(analyzer="word",
								tokenizer=None,
								preprocessor=None,
								stop_words=None,
								max_features=5000)
								
	data_features = vectorizer.fit_transform(clean_data)
	data_features = data_features.toarray()
	
	return data_features
	
def RandomForestModel(train_data_features,train):
	print("Training the Random Forest Classifier...")	
	model = RandomForestClassifier(n_estimators = 100, verbose=True)
	model = model.fit(train_data_features, train["sentiment"])
	#save Model	
	filename = 'models/RandomForest_model.sav'
	joblib.dump(model, filename)
	return model
	
def Decision_Tree_Classifier(train_data_features,train):
	model = tree.DecisionTreeClassifier()
	model = model.fit(train_data_features,train["sentiment"])
	#save Model	
	filename = 'models/DecisionTreeClassifier_model.sav'
	joblib.dump(model, filename)
	return model
	
def train_model(train):
	#Clean Training Data
	print('Cleaning and Parsing the training set movie reviews...\n')
	clean_train_reviews = []	
	clean_train_reviews = sentence2word(train)

	#Creating Bag of Words
	train_data_features = data_vectorize(clean_train_reviews)
	
	#Training the Classifier
	model = RandomForestModel(train_data_features,train)
	#model = Decision_Tree_Classifier(train_data_features,train)		

def test_model(test):
	#Loading Model
	filename = 'models/RandomForest_model.sav'
	model = joblib.load(filename)
	
	#Formating the testing data
	print("Cleaning and parsing the test set movie reviews...\n")
	
	clean_test_reviews = []	
	clean_test_reviews = sentence2word(test)
	
	test_data_features = data_vectorize(clean_test_reviews)
	
	#Predict reviews in test data
	print("Predicting Test Labels...")
	result = model.predict(test_data_features)
	output = pd.DataFrame(data={"id":test["id"], "sentiment":result})
	output.to_csv(os.path.join(os.path.dirname(__file__), 'Result', 'Bag_of_Words_model.csv'), index=False, quoting=3)	
	print("Wrote results of Bag_of_Words_model.csv")
	


if __name__ == '__main__':
	foldername = 'Dataset'
	filename_type = 'labeledTrainData.tsv'
	train = load_data(foldername, filename_type)
	filename_type = 'testData.tsv'
	test = load_data(foldername, filename_type)
	
	train_model(train)
	test_model(test)
	