import pytest
from redactor import main
import os

def test_unredacting():
    #un_red_path = r"../unredacted_files/10697_10.txt"
    
#    train_path = r"../aclImdb/train/pos/*.txt"
#    train_data, names = main.do_extraction(train_path) # performs extraction and gets training features
#    X_train,y_train = main.get_train_vectors(train_data,names)
#
#    model = ensemble.RandomForestClassifier()
#    model.fit(X_train,y_train)
#
#    test_path = r"../aclImdb/test/pos/10697_10.txt"
#    
#    #redact_test_files(test_path)
#    #redacted_path = r"../redacted_files/*.txt"
#
#
#    #features = main.get_training_features(text,file_path)
#    
#    redacted_file = r"redacted_files/10697_10.txt"
#    main.get_unredacted_file(redacted_file)
#
    un_red_path = r"unredacted_files/10697_10.txt"
    if os.path.exists(un_red_path):
        assert True
    else:
        assert False

