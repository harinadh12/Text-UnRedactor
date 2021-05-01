import spacy
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
from sklearn import ensemble
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
import os
from statistics import mode
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import glob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

nlp = spacy.load('en_core_web_md')
sid = SentimentIntensityAnalyzer()
nltk.download('stopwords')
dict_vectorizer = DictVectorizer()

def do_extraction(folder):
    train_data = []

    for each_file in glob.glob(folder)[:1000]:
        text = open(each_file,'r').read()
        #if 'robin' in text.lower():
        features = get_training_features(text,each_file)
        train_data.extend(features)
    
    names = []
    for each_feat in train_data:
        names.append(each_feat['name'])
        del each_feat['name']

    return train_data, names

def get_train_vectors(train_data, names):

    #dict_vectorizer = DictVectorizer()
    X_train = dict_vectorizer.fit_transform(train_data).toarray()
    y_train = np.asarray(names)
    print(y_train.shape)
    return X_train,y_train


def get_training_features(text,file_name):

    doc = nlp(text)
    word_list = [w for w in text.split() if w not in stopwords.words('english')]
        
    feature_list = []
    for sent in doc.sents:
        pos_words = []
        neg_words = []

        for word in sent.text.split():
            if (sid.polarity_scores(word)['compound']) >= 0.5:
                pos_words.append(word)
            elif (sid.polarity_scores(word)['compound']) <= -0.5:
                neg_words.append(word)
        lemmatizer = WordNetLemmatizer()
        pos_words = [lemmatizer.lemmatize(word.lower(),pos="a") for word in pos_words]
        neg_words = [lemmatizer.lemmatize(word.lower(),pos="a") for word in neg_words]
        pos_words = list(set(pos_words))
        neg_words = list(set(neg_words))

        for ent in sent.ents:
        
            if ent.label_ == "PERSON":
                feature_dict = {}
                feature_dict['name_length'] = len(ent.text)
                feature_dict['name'] = ent.text
                #feature_dict['no_spaces'] = len(ent.text.split())-1
                feature_dict['user_rating'] = int(os.path.basename(file_name).split('_')[-1][:-4])
                if int(os.path.basename(file_name).split('_')[-1][:-4]) >= 5:
                    feature_dict['sntmt_words'] = pos_words
                elif int(os.path.basename(file_name).split('_')[-1][:-4]) < 5:
                    feature_dict['sntmt_words'] = neg_words
                feature_list.append(feature_dict)
        
    return feature_list


def get_unredacted_file(red_file):
    
    # for red_file in glob.glob(redact_folder):
        
    text = open(red_file,'r').read()
    doc = nlp(text)
    

    for sent in doc.sents:
        pos_words = []
        neg_words = []

        for word in sent.text.split():
            if (sid.polarity_scores(word)['compound']) >= 0.5:
                pos_words.append(word)
            elif (sid.polarity_scores(word)['compound']) <= -0.5:
                neg_words.append(word)
        lemmatizer = WordNetLemmatizer()
        pos_words = [lemmatizer.lemmatize(word.lower(),pos="a") for word in pos_words]
        neg_words = [lemmatizer.lemmatize(word.lower(),pos="a") for word in neg_words]
        pos_words = list(set(pos_words))
        neg_words = list(set(neg_words))


        for ent in sent.text.split():
            
            if '\u2588' in ent:
                
                feature_dict = {}
                feature_dict['name_length'] = len(ent)
                
                feature_dict['user_rating'] = int(os.path.basename(red_file).split('_')[-1][:-4])
                if int(os.path.basename(red_file).split('_')[-1][:-4]) >= 5:
                    feature_dict['sntmt_words'] = pos_words
                elif int(os.path.basename(red_file).split('_')[-1][:-4]) < 5:
                    feature_dict['sntmt_words'] = neg_words
                feature_list.append(feature_dict)
                X_test = dict_vectorizer.transform(feature_dict).toarray()
                
               
               #print(model.predict(X_test))
                probs = model.predict_proba(X_test)
                for i in range(len(probs)):
                    top_5_idx = np.argsort(probs[i])[-5:]
                    top_5_values = [y_train[i] for i in top_5_idx]
                    text += "\n Top 5 predictions for the text at position {} are {} \n".format(i+1, top_5_values)

    un_red_file = '../unredacted_files/'+os.path.basename(red_file)
    with open(un_red_file,'w') as f:

        f.write(text)
        f.close()


def redact_test_files(test_folder):
    
    for each_file in glob.glob(test_folder)[:10]:
        text = open(each_file,'r').read()
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                text = text.replace(ent.text,'\u2588' * len(ent.text))
                
        redact_file = "../"+ 'redacted_files/' + os.path.basename(each_file)
        file = open(redact_file,"w")
        file.write(text)
        file.close()

    
if __name__ == '__main__':
    
    train_path = r"../aclImdb/train/pos/*.txt"
    train_data, names = do_extraction(train_path) # performs extraction and gets training features
    X_train,y_train = get_train_vectors(train_data,names)

    model = ensemble.RandomForestClassifier()
    model.fit(X_train,y_train)
    
    test_path = r"../aclImdb/test/pos/*.txt"
    redact_test_files(test_path)
    redacted_path = r"../redacted_files/*.txt"
    
    for red_file in glob.glob(redacted_path):
        get_unredacted_file(red_file)

    
