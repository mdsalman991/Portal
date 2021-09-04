import pandas as pd
import sys
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_union
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
import emoji
import re
from bs4 import BeautifulSoup
import itertools
import pickle
import util
from pymongo import MongoClient
from pprint import pprint

# Create df of the question + answer + comment and preprocess them
def Create_df():
    client = MongoClient(port=27017)
    db=client.portal
    result = db.discussions.find({})
    df_new = pd.DataFrame(columns=['comment_body','Disagreement','Partial','Agreement'])
    for d in result:
        ques = d['question']+" "
        for i in d['answers']:
            result_ans = db.answers.find({"_id":i})
            for a in result_ans:
                ans = a['answer']
                com = " "
                for j in a['comments']:
                    result_com = db.comments.find({"_id":j})
                    for c in result_com:
                        comment = c['comment']
                        com = com + comment + " "
                        if(c['annotate'] != "None"):
                            que_ans_com = ques+ans+com
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
                            if(c['annotate'] == "Disagreement"):
                                df_new = df_new.append({'comment_body': que_ans_com,
                                                        'Disagreement':1,
                                                        'Partial':0, 
                                                        'Agreement':0},ignore_index=True )
                            elif(c['annotate'] == "Partial"):
                                df_new = df_new.append({'comment_body': que_ans_com,
                                                        'Disagreement':0,
                                                        'Partial':1,
                                                        'Agreement':0},ignore_index=True )
                            else:
                                df_new = df_new.append({'comment_body': que_ans_com,
                                                        'Disagreement':0,
                                                        'Partial':0,
                                                        'Agreement':1},ignore_index=True )
    return df_new




df = pd.read_excel('segregated_formdata_comment.xls')

df1 = df
# Load prev dataframe
for i in range(0,507):
    string = df1.iloc[i,11]

    string = BeautifulSoup(string,features="html5lib").get_text()
    string = string.replace('\x92',"'")
    string = ' '.join(re.sub("[\.\,\!\?\:\;\-\=]", " ", string).split())
    CONTRACTIONS = util.load_dict_contractions()
    string = string.replace("’","'")
    words = string.split()
    reformed = [CONTRACTIONS[word] if word in CONTRACTIONS else word for word in words]
    string = " ".join(reformed)

    string = ''.join(''.join(s)[:2] for _, s in itertools.groupby(string))

    SMILEY = util.load_dict_smileys()  
    words = string.split()
    reformed = [SMILEY[word] if word in SMILEY else word for word in words]
    string = " ".join(reformed)
    
    #Deal with emojis
    string = emoji.demojize(string)

    string = string.replace(":"," ")
    string = ' '.join(string.split())
    df1.iloc[i,11] = string

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    stop_words='english',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 2),
    max_features=30000)
char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    ngram_range=(1, 4),
    max_features=30000)

vectorizer = make_union(word_vectorizer,char_vectorizer)

y = df1[['Disagreement','Partial','Agreement']]
X = df1.drop(y,axis = 1)

#print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3  , random_state=42)
df_tr= X_train['comment_body']
df_test = X_test['comment_body']

vectorizer.fit(df_tr)

train_features = vectorizer.transform(df_tr)
test_features = vectorizer.transform(df_test)
classname = ['Disagreement','Partial','Agreement']
#preds = np.zeros((len(df_test), len(classname))) 
preds = pd.DataFrame.from_dict({'id': X_test['comment_id']})
#print(preds.shape)

m = 0
for class_name in classname:
    train = y_train[class_name]
    clf = RandomForestClassifier(n_estimators=500, 
                                 max_leaf_nodes=None,
                                 max_features=0.6, 
                                 min_samples_leaf=6, 
                                 random_state=42)
    clf = clf.fit(train_features, train)
    m+=1
    preds[class_name] = clf.predict_proba(test_features)[:,1]


    

p = preds[['Disagreement','Partial','Agreement']]
p = p.eq(p.where(p != 0).max(1), axis=0).astype(int)
from sklearn.metrics import accuracy_score,f1_score
from sklearn.metrics import confusion_matrix 
print("***********OLD MODEL**************")
print("Confusion Matrix:")
print(confusion_matrix(y_test.values.argmax(axis = 1), p.values.argmax(axis=1)))
print("Accuracy score: ",accuracy_score(y_test,p))
print("F1 score: ",f1_score(y_test,p,average='weighted'))

# add db train data to df1
df_new = Create_df()
df1 = df1.append(df_new, ignore_index=True)

y = df1[['Disagreement','Partial','Agreement']]
X = df1.drop(y,axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3  , random_state=42)
df_tr= X_train['comment_body']
df_test = X_test['comment_body']
# print (y)
vectorizer.fit(df_tr)
pickle.dump(vectorizer,open("Retrained/vectorizer_tfidf_dummy.pkl",'wb'))    

train_features = vectorizer.transform(df_tr)
test_features = vectorizer.transform(df_test)
classname = ['Disagreement','Partial','Agreement']
preds = pd.DataFrame.from_dict({'id': X_test['comment_id']})

m = 0
for class_name in classname:
    train = y_train[class_name]
    train=train.astype('int')
    clf = RandomForestClassifier(n_estimators=500, 
                                 max_leaf_nodes=None,
                                 max_features=0.6,
                                 min_samples_leaf=6, 
                                 random_state=42)
    clf = clf.fit(train_features, train)
    m+=1
    pickle.dump(clf,open("Retrained/"+str(class_name)+str(m)+"_dummy.pkl",'wb'))    
    preds[class_name] = clf.predict_proba(test_features)[:,1]

p = preds[['Disagreement','Partial','Agreement']]
p = p.eq(p.where(p != 0).max(1), axis=0).astype(int)
print("***********NEW MODEL**************")
print("Confusion Matrix:")
print(confusion_matrix(y_test.values.argmax(axis = 1), p.values.argmax(axis=1))) 
print("Accuracy score: ",accuracy_score(y_test.values.argmax(axis = 1), p.values.argmax(axis=1)))
print("F1 score: ",f1_score(y_test.values.argmax(axis = 1), p.values.argmax(axis=1),average='weighted'))

# To get minimum agreement score 
train = y_train['Agreement'].values.tolist()
clf = RandomForestClassifier(n_estimators=500, 
                             max_leaf_nodes=None,
                             max_features=0.6, 
                             min_samples_leaf=6, 
                             random_state=42)
clf = clf.fit(train_features, train)
preds = clf.predict_proba(train_features)[:,1]
minimum = 1
for i in range(len(y_train['Agreement'])):
    if(train[i] == 1):
        if(preds[i]<minimum):
            minimum = preds[i]
print("min_agree_prob, = ",minimum)

# to get abs(avg(partial-agreement))
train = y_train['Partial'].values.tolist()
clf1 = RandomForestClassifier(n_estimators=500, 
                              max_leaf_nodes=None,
                              max_features=0.6, 
                              min_samples_leaf=6, 
                              random_state=42)
clf1 = clf1.fit(train_features, train)
preds1 = clf1.predict_proba(train_features)[:,1]
total = 0
for i in range(len(y_train['Partial'])):
    total += abs(preds[i]-preds1[i])
print("avg_of_diff = ",total/len(y_train['Partial']))