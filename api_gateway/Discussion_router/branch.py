import pandas as pd
import re
import math
import pprint
import nltk
# nltk.download('stopwords')
import nltk.corpus
from nltk.corpus import nps_chat
from nltk.corpus import stopwords
import numpy as np
from nltk import tokenize
# import textstat
import pickle
# import spacy
from nltk import tokenize
import pickle
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter
from textblob import TextBlob
import string
import util
import emoji
import itertools
# Function called by the userdata script to get Agreement/disagreement
def getResult(discussion,answers,comments):
	def remove(list): 
		pattern = '[0-9]'
		list = [re.sub(pattern, '', i) for i in list] 
		return list
    #function to get some useful feature iterating through comments
	def getDetails():
		comms = "" 
		no_unique_people = set() 
		no_unique_people.add(discussion['userid'])
		no_unique_people.add(answers['userid'])
		is_QuestionerBack = 0
		depth = 2
		is_LastCommentQues = 0
		no_intermediateQuestion = 0 
		for c in comments:
			comms += (c['comment'] + " ")
			if(c['userid'] == discussion['userid']):
				is_QuestionerBack = 1
			no_unique_people.add(c['userid'])
			depth += 1
			if('?' in c['comment']):
				no_intermediateQuestion += 1
				if(c == comments[-1]):
					is_LastCommentQues=1
		return comms,len(no_unique_people),is_QuestionerBack,depth,is_LastCommentQues,no_intermediateQuestion,no_unique_people

	# convert test to dataframe of appropriate form
	comms, no_unique_people, is_QuestionerBack, depth, is_LastCommentQues, no_intermediateQuestion,unique_people = getDetails()
	question_body = discussion['question']
	answer_text = answers['answer']
	no_upvotes = 0
	# TO use only the votes that are given by participants of the thread (To avoid spam to break the model)
	for i in answers['votes']:
		if (str(i) in unique_people):
			no_upvotes+=1
	for i in discussion['votes']:
		if (str(i) in unique_people):
			no_upvotes+=1
	is_TAVerified = int(answers['TA_verified'])

	

	que_ans_com =  question_body + " " + answer_text + " " + comms
	que_ans_com = que_ans_com.replace('\x92',"'")
	que_ans_com = ' '.join(re.sub("[\.\,\!\?\:\;\-\=]", " ", que_ans_com).split())
	CONTRACTIONS = util.load_dict_contractions()
	que_ans_com = que_ans_com.replace("’","'")
	words = que_ans_com.split()
	reformed = [CONTRACTIONS[word] if word in CONTRACTIONS else word for word in words]
	que_ans_com = " ".join(reformed)

	que_ans_com = ''.join(''.join(s)[:2] for _, s in itertools.groupby(que_ans_com))

	SMILEY = util.load_dict_smileys()  
	words = que_ans_com.split()
	reformed = [SMILEY[word] if word in SMILEY else word for word in words]
	que_ans_com = " ".join(reformed)
	
	#Deal with emojis
	que_ans_com = emoji.demojize(que_ans_com)

	que_ans_com = que_ans_com.replace(":"," ")
	que_ans_com = ' '.join(que_ans_com.split())

	final = {
			'que_ans_com' : [que_ans_com],
			'depth' : [depth],
			'no_unique_people':[no_unique_people],
			'no_upvotes': [no_upvotes],
			'is_TAVerified' : [is_TAVerified], 'is_LastCommentQues' : [is_LastCommentQues], 'is_QuestionerBack' : [is_QuestionerBack],
		   'no_intermediateQuestion':[no_intermediateQuestion],'Disagreement_prob':[0], 'Partial_prob':[0],
		   'Agreement_prob':[0]
	}

	# print(final)

	feature_set = pd.DataFrame.from_dict(final)
	feature_set['polarity'] = feature_set['que_ans_com'].apply(lambda x: TextBlob(x).sentiment.polarity)
	feature_set['upper_count'] = feature_set['que_ans_com'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))
	feature_set['no_intermediateQuestion/no_unique_people'] = feature_set.apply(lambda x: x['no_intermediateQuestion'] / float(x['no_unique_people'] if x['no_unique_people']!=0 else 0.0), axis=1)
	feature_set['punc_count'] = feature_set['que_ans_com'].apply(lambda x: len([i for i in x if i in string.punctuation]))
	feature_set['title_case'] = feature_set['que_ans_com'].apply(lambda x: len([word for word in x.split() if word.istitle()]))

	# Preprocess
	stop_words = set(stopwords.words('english')) 
	tokenizer = RegexpTokenizer(r'\w+')
	documents = []
	for i in range(len(feature_set)): 
		s = feature_set.loc[i][0].lower()
		word_tokens = tokenizer.tokenize(s)
		filtered_sentence = [w for w in word_tokens if not w in stop_words]
		filtered_sentence = remove(filtered_sentence)
		filtered_sentence = [x for x in filtered_sentence if x]
		s = ' '.join([str(elem) for elem in filtered_sentence]) 
		documents.append(s)
	preprocessed_comments = []
	s = que_ans_com.lower()
	word_tokens = tokenizer.tokenize(s)
	filtered_sentence = [w for w in word_tokens if not w in stop_words]
	filtered_sentence = remove(filtered_sentence)
	filtered_sentence = [x for x in filtered_sentence if x]
	s = ' '.join([str(elem) for elem in filtered_sentence]) 
	preprocessed_comments.append(s)
	# print(preprocessed_comments)
	tfidf = pd.read_pickle('vectorizer_tfidf_fulldata.pkl')
	d = tfidf.transform(preprocessed_comments)
	feature_set['que_ans_com'] = documents

	# Calculate invidual probability
	m1 = pd.read_pickle('Disagreement1_full_df.pkl')
	y1 = m1.predict_proba(d)
	feature_set['Disagreement_prob'] = [y1[0][1]]
	m2 = pd.read_pickle('Partial2_full_df.pkl')
	y2 = m2.predict_proba(d)
	feature_set['Partial_prob'] = [y2[0][1]]
	m3 = pd.read_pickle('Agreement3_full_df.pkl')
	y3 = m3.predict_proba(d)
	feature_set['Agreement_prob'] = [y3[0][1]]
	# print(feature_set['Disagreement_prob'])
	# print(feature_set['Partial_prob'])
	# print(feature_set['Agreement_prob'])
	# dense tfidf
	tfidf_dense = pd.read_pickle('vectorizer.pk')
	l = tfidf_dense.transform(documents)
	feature_names = tfidf_dense.get_feature_names()
	dense = l.todense()
	denselist = dense.tolist()
	df = pd.DataFrame(denselist, columns=feature_names)

	result = pd.concat([feature_set, df], axis=1)

	# Model test
	model = pd.read_pickle('branchlevel_model.pkl')
	y = model.predict_proba(result.iloc[:,1:])
	if(y[0][2]>y[0][0] and y[0][2]>y[0][1]):
		result = 0
	else:
		result = 1
	return result
