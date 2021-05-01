import pytest
from redactor import main

file_path = r"aclImdb/test/pos/10697_10.txt"

def test_extraction():
    train_data,names = main.do_extraction(file_path)
    assert len(names)>=0

