
##                                                 THE UNREDACTOR 
AUTHOR : _**HARINADH APPIDI**_

In this Project '.txt' files of IMDB movie reviews where Person names are redacted are given as input and top 5 matches of the person names are predicted and written to unredacted files folder.  

To run the project navigate to the redactor folder and run 'pipenv run python main.py' at the command line.  
To install the required packages run 'pipenv install' from the main folder.  

All the Unredacted files are written to unredacted_files folder and redacted files are written in redacted_files folder.  
In this project only 1000 positive training files are taken for training and 10 testing files are used for testing.  
For any changes please refer to main.py file in the respective functions.  

The following methods are written in main.py file and performs the below functions.  

**do_extraction(folder)** - This function takes trainig files as input and performs extraction , performs feature extraction and returns training features, corresponding label.  
**get_training_features(text,file_name)** - This method takes extracted text, file name and returns features list of all the person names in the file.  
**get_train_vectors(train_data, names)** -  This method takes in training features, corresponding labels and converts to vectorized format of X_train, y_train respectively.  
**redact_test_files(test_folder)** - This method takes in a text file and redacts Person names using spacy entity recognition.  
**get_unredacted_file(red_file)** - This method takes redacted file and performs prediction using Random Forest classifier model to predict top 5 names for the redacted names.  

In this project for feature extraction, I used name length, user rating and sentiment words as features.  
Sentiment words are identified by performing the sentiment analysis of the sentence where person name is identified. This caused to increase the dimensionality of the feature vector as the number of features increase based on the number of sentiment words.  
So, For a review rated atleast 5/10 i take positive words and review rated less than 5/10 i take negative words as features which are converted to feature vector representation using dict vectorizer.  
Sentiment words are used as features based on the assumption that movie with similar ratings get the same kind of appreciation words like great, wonderful,loved,like etc..  for positive cases and hated,worst,killed,disgusting etc.. for negative scenarios.  
This is based on the assumption that movie goes either into positive (>=5) cycle or negative(<5) movie list.  
This may not be true as everyone doesnt give positive ratings or negative ratings as some might like the movie and some others might not have liked the movie.  

Model training is performed using Random Forest classifier so as to get the predicition probabilities and multiple label predictions.  
Model is tested with one sample file used for training for testing and all the labels are predicted correctly.  
However due to computational limitations of increase in feature vector size with more training data, Accurate results are not predicted for test data whch model has never seen in the training.
 
I tried another approach to identify the context words using spacy vectors which works based on word2vec model.  
However due to limitations of some of the first names or last names being not found in spacy corpus, I couldn't get a vector representation for those words.  
This code is coded in [other file](./redactor/other.py) of redactor folder.

####  TEST CASES  
Four test files are coded for testing different test files.
  * 1. test_extraction.py - This file is used for testing do_extraction method. It asserts True if atleast one feature(person name) is identified in the moview review file.  
  * 2. test_redacting.py  -  This file is used for testing redaction process . It asserts True if redacted file for input file is found in redacted_files folder.   
  * 3. test_training.py - This file is used for testing training process of the movie review files. It asserts True if atleast one feature is returned from the      get_training_features method.  
  * 4. test_unredacting.py - This file is used for testing get_unredacted_file method. It asserts True if unredacted file is found in the unredacted_files folder for the iput file.

To run the test cases run '_pipenv run python setup.py test_' in the main folder.  

References:

  * 1. https://www.nltk.org/api/nltk.sentiment.html
  * 2. https://spacy.io/usage/linguistic-features
  * 3. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html
  * 4. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
  * 5. https://applied-language-technology.readthedocs.io/en/latest/notebooks/part_iii/04_embeddings_continued.html

