import spacy
import sklearn
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import pandas as pd
import nltk
import re
import glob


nlp = spacy.load('en_core_web_md')



def doextraction(file_path):
    all_files = glob.glob(file_path)
    all_data = []
    new_files = []
    for each_file in all_files[:100]:
        try:
            text = open(each_file,'r').read()
            text = re.sub('[^a-zA-Z\' ]','',text)
            all_data.append(text)
            new_files.append(each_file)
        except:
            continue
    #print(all_data)
    return all_data, new_files

def redact_testdata(test_data):
    redact_data = []
    doc_vec = []
    for text in test_data:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                text = text.replace(ent.text, "\u2588" * len(ent.text))
        redact_data.append(text)
        #doc_vec.append(doc.vector)
    #df = pd.DataFrame(list(zip(redact_data,doc_vec)))
    #print(df)
    return redact_data

def redact_person(train_data):
    try:
        if isinstance(train_data,list):
            redact_vectors = []
            redact_data = []
            redact_names =[]
            redact_list = []
            for text in train_data:
                doc = nlp(text)
                for ent in doc.ents:
                    if ent.label_ == "PERSON":
                        #text = text.replace(ent.text, "\u2588" * len(ent.text))
                        #redact_vectors.append(ent.vector)
                        redact_names.append(ent.text)
                        e1 = np.append(ent.vector,len(ent.text))
                        e2 = np.append(e1, len(ent.text.split()))
                        redact_vectors.append(e2)

                        #print(e2.shape)
                        #redact_list.append((e2, ent.text))
                redact_data.append(text)
            redact_list = list(zip(redact_vectors,redact_names))
            #print(pd.DataFrame(redact_list))
            #print([*redact_list])

        return redact_list,redact_names

    except Exception as e:
        print("Exception in Redact Person Method")
        raise e

def cosine_angle(x,y):
    return (np.dot(x,y))/(np.linalg.norm(x)*np.linalg.norm(y))


def get_test_features(redacted_data):
    vector_list = []
    for each_file in redacted_data:
        doc = nlp(each_file)
        for token in doc:
            if "\u2588" in token.text:
                #print (token.vector)
                e1 = np.append(token.vector,len(token.text))
                e2 = np.append(e1,len(token.text.split()))
                vector_list.append(e2)
    
    return vector_list

if __name__ == "__main__":

    train_path = r"../aclImdb/train/pos/*.txt"

    train_data,train_file_names = doextraction(train_path)
    
    redact_list, redact_data = redact_person(train_data)

    #print(redact_list[0][0])    
    #doc = nlp("oliver")
    #(np.dot(doc[0].vector,redact_list[0][0]))/(np.linalg.norm(doc[0].vector)*np.linalg.norm(redact_list[0][0]))
    #print(t)

    test_path = r"../aclImdb/test/pos/*.txt"
    test_data,test_file_names = doextraction(test_path)
    redact_test = redact_testdata(test_data)
    vector_list = get_test_features(redact_test)

    


    #print("***********",test_file_names)
    #print(redact_data)
    #print(redact_test)
#    print("Except Angle finding")
#    for train_vec in redact_list:
#         for each_vec in vector_list:
#             if cosine_angle(train_vec[0],each_vec) >= 0.50:
#                 #if train_vec[0][-2] == each_vec[-2]:
#                 print(train_vec[1])
# 

        #break




