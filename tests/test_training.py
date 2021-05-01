import pytest
from redactor import main

file_path = r"aclImdb/test/pos/10697_10.txt"


def test_training_features():
    
    text = open(file_path,'r').read()

    features = main.get_training_features(text,file_path)
    assert  len(features) >=0

