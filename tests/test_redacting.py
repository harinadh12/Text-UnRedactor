import pytest
from redactor import main
import os

file_path = r"aclImdb/test/pos/10697_10.txt"
def test_redact_files():

    main.redact_test_files(file_path)
    redact_file_path = r"redacted_files/10697_10.txt"
    assert os.path.exists(redact_file_path)



