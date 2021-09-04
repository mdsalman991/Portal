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
from Discussion_router import util
import emoji
import itertools

# Function called by the server to get Agreement/disagreement
def getResult(test,index):
    def remove(list): 
        pattern = '[0-9]'
        list = [re.sub(pattern, '', i) for i in list] 
        return list
    #function to get some useful feature iterating through comments
    def getDetails(d):
    	comments = "" 
    	no_unique_people = set() 
    	no_unique_people.add(d['userid'])
    	no_unique_people.add(d['answers'][index]['userid'])
    	is_QuestionerBack = 0
    	depth = 2
    	is_LastCommentQues = 0
    	no_intermediateQuestion = 0 
    	for c in d['answers'][index]['comments']:
    		comments += (c['comment'] + " ")
    		if(c['userid'] == d['userid']):
    			is_QuestionerBack = 1
    		no_unique_people.add(c['userid'])
    		depth += 1
    		if('?' in c['comment']):
    			no_intermediateQuestion += 1
    			if(c == d['answers'][index]['comments'][-1]):
    				is_LastCommentQues = 1
    	return comments,len(no_unique_people),is_QuestionerBack,depth,is_LastCommentQues,no_intermediateQuestion

    # convert test to dataframe of appropriate form
    question_body = test['question']
    answer_text = test['answers'][index]['answer']
    no_upvotes = len(test['answers'][index]['votes']) + len(test['votes'])
    is_TAVerified = int(test['answers'][index]['TA_verified'])

    comments, no_unique_people, is_QuestionerBack, depth, is_LastCommentQues, no_intermediateQuestion = getDetails(test)

    que_ans_com =  question_body + " " + answer_text + " " + comments
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
            'is_TAVerified' : [is_TAVerified], 
            'is_LastCommentQues' : [is_LastCommentQues], 
            'is_QuestionerBack' : [is_QuestionerBack],
            'no_intermediateQuestion':[no_intermediateQuestion],
            'Disagreement_prob':[0], 
            'Partial_prob':[0],
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
    tfidf = pd.read_pickle('Discussion_router/vectorizer_tfidf_fulldata.pkl')
    d = tfidf.transform(preprocessed_comments)
    feature_set['que_ans_com'] = documents

    # Calculate invidual probability
    m1 = pd.read_pickle('Discussion_router/Disagreement1_full_df.pkl')
    y1 = m1.predict_proba(d)
    feature_set['Disagreement_prob'] = [y1[0][1]]
    m2 = pd.read_pickle('Discussion_router/Partial2_full_df.pkl')
    y2 = m2.predict_proba(d)
    feature_set['Partial_prob'] = [y2[0][1]]
    m3 = pd.read_pickle('Discussion_router/Agreement3_full_df.pkl')
    y3 = m3.predict_proba(d)
    feature_set['Agreement_prob'] = [y3[0][1]]
    # dense tfidf
    tfidf_dense = pd.read_pickle('Discussion_router/vectorizer.pk')
    l = tfidf_dense.transform(documents)
    feature_names = tfidf_dense.get_feature_names()
    dense = l.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)

    result = pd.concat([feature_set, df], axis=1)

    # Branch Model test
    model = pd.read_pickle('Discussion_router/branchlevel_model.pkl')
    y = model.predict_proba(result.iloc[:,1:])
     
    avg_of_dif = 0.364                   #To be changed based on the training set
    min_of_agree = 0.472                 #To be changed based on the training set
    if(depth<=4):
        return 0,0
    else:
        if(y3[0][1]>y2[0][1] and y3[0][1]>y1[0][1]):
            result = 0
        elif(y3[0][1]<min_of_agree and abs(y2[0][1]-y3[0][1])>avg_of_dif):
            result = 1
        else :
            result = 0

    return result,int(np.argmax(y[0]))




