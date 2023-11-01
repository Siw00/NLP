
# 2 In Google Colab, develop a Simple Python Program to count the number of words in a paragraph and user should be promoted to enter a paragraph through a file: Note: The paragraph should not be hand-typed.
# AIM: To develop a simple python program to count the number of words in a paragrap
from google.colab import files
VAR_input_file = input("Enter the path to the file of x.txt:")
uploaded = files.upload()
VAR_input_file

VAR_Count = 0
with open(VAR_input_file,'r') as j:
  for VAR_sent in j:
    VAR_tokn = VAR_sent.split()
    VAR_Count+=len(VAR_tokn)
print (VAR_Count)

# 3A In Google Colab, develop a Simple Python Program to reverse the sentence and count the length of each word in a sentence and then store them in an list stating with the word which has highest string length
# AIM:To develop a simple python program to reverse the sentence and count the length of the each word in a sentence and the n store in an list stating with the world which has highest string length
VAR_input= "This is Natural Language Processing"
VAR_Y = VAR_input.split()[::-1]
VAR_Y
VAR_len = []
for x1 in VAR_Y:
  VAR_len.append(len(x1))
VAR_res = {}
for i,j in zip(VAR_Y,VAR_len):
  VAR_res[i] = j
VAR_res
import collections
VAR_S = collections.OrderedDict(sorted(VAR_res.items(), key=lambda x: x[1], reverse=True))
VAR_S
VAR_words_maxlength = VAR_S.keys()

print("words in decreasing order of length are:",VAR_words_maxlength)

# 4A In Google Colab, develop a Simple Python Program for Creating Arguments,Mutable Arguments and Accepting Variable Arguments
import nltk

nltk.download("punkt")
nltk.download("stopwords")
import re
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import collections
from nltk.tokenize import word_tokenize
var_input= """transformers consists of encoder and decoder machine learning is the field of study that gives computers the capability to learn
ML is one of the most exciting technologies that one would have ever come across. As it is evident from the name,
it gives the computer that makes it more similar to humans: The ability to learn.
Machine learning is actively being used today, perhaps in many more places than one would expect.
It can be any unprocessed fact, value, text, sound, or picture that is not being interpreted and analyzed.
Data is the most important part of all Data Analytics, Machine Learning, Artificial Intelligence.
Without data, we can’t train any model and all modern research and automation will go in vain.
Big Enterprises are spending lots of money just to gather as much certain data as possible.
The part of data that is used to do a frequent evaluation of the model, fit on the training dataset along with
improving involved hyperparameters (initially set parameters before the model begins learning).
This data plays its part when the model is actually training.
Once our model is completely trained, testing data provides an unbiased evaluation. When we feed in the inputs of Testing data,
our model will predict some values(without seeing actual output). After prediction, we evaluate our model by comparing it
with the actual output present in the testing data. This is how we evaluate and see how much our model has learned from
the experiences feed in as training data, set at the time of training."""
stopwords= nltk.corpus.stopwords.words('english')
print('the stopwords are:',stopwords)
var_input= word_tokenize(var_input)
var1= [i for i in var_input if i.lower() not in stopwords]
var_tokenized=''.join(var1)
print(var_tokenized)

#4b Perform Stemming and Lemmatization in the given text and remove  stopwords.
import nltk

nltk.download("punkt")
nltk.download("stopwords")
import re
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import collections
from nltk.tokenize import word_tokenize
var_input= """transformers consists of encoder and decoder machine learning is the field of study that gives computers the capability to learn
ML is one of the most exciting technologies that one would have ever come across. As it is evident from the name,
it gives the computer that makes it more similar to humans: The ability to learn.
Machine learning is actively being used today, perhaps in many more places than one would expect.
It can be any unprocessed fact, value, text, sound, or picture that is not being interpreted and analyzed.
Data is the most important part of all Data Analytics, Machine Learning, Artificial Intelligence.
Without data, we can’t train any model and all modern research and automation will go in vain.
Big Enterprises are spending lots of money just to gather as much certain data as possible.
The part of data that is used to do a frequent evaluation of the model, fit on the training dataset along with
improving involved hyperparameters (initially set parameters before the model begins learning).
This data plays its part when the model is actually training.
Once our model is completely trained, testing data provides an unbiased evaluation. When we feed in the inputs of Testing data,
our model will predict some values(without seeing actual output). After prediction, we evaluate our model by comparing it
with the actual output present in the testing data. This is how we evaluate and see how much our model has learned from
the experiences feed in as training data, set at the time of training."""
stopwords= nltk.corpus.stopwords.words('english')
print('the stopwords are:',stopwords)
var_input= word_tokenize(var_input)
var1= [i for i in var_input if i.lower() not in stopwords]
var_tokenized=''.join(var1)
print(var_tokenized)
from nltk.stem import PorterStemmer
obj_st= PorterStemmer()
var_stemmed= obj_st.stem(var_tokenized)
print("output after stemming is:",var_stemmed)

# 5b Extract Noun, Pronoun, Verbs and Adjectives from the given text.
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Download NLTK data (if not already downloaded)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Define the text containing the definition of machine learning
text = "Machine learning is a subset of artificial intelligence that involves the use of algorithms and statistical models to enable computer systems to learn from and make predictions or decisions without being explicitly programmed. Machine learning is used in various applications, including image and speech recognition, recommendation systems, and natural language processing."

# Tokenize the text into words
words = word_tokenize(text)

# Perform part-of-speech tagging to identify nouns, pronouns, verbs, and adjectives
tagged_words = pos_tag(words)

# Initialize empty lists for different parts of speech
nouns = []
pronouns = []
verbs = []
adjectives = []

# Iterate through the tagged words and categorize them
for word, tag in tagged_words:
    if tag.startswith('N'):  # Nouns
        nouns.append(word)
    elif tag == 'PRP' or tag == 'PRP$':  # Pronouns
        pronouns.append(word)
    elif tag.startswith('V'):  # Verbs
        verbs.append(word)
    elif tag.startswith('JJ'):  # Adjectives
        adjectives.append(word)

# Print the extracted words
print("Nouns:", nouns)
print("Pronouns:", pronouns)
print("Verbs:", verbs)
print("Adjectives:", adjectives)


# 8 Create a Question-Answering system from given context
pip3 install transformers
from transformers import pipeline as QA
obj_model = QA("question-answering",model="distilbert-base-cased-distilled-squad")
var_ques="what is the model for question generation"
var_para = "The BERT (Bidirectional Encoder Representations from Transformers) model, specifically the 'bert-base-cased-distilled' version, is a pre-trained model that has been fine-tuned on the Stanford Question Answering Dataset (SQuAD). BERT is known for its ability to understand the context of text and is widely used in various natural language processing tasks, including question-answering. It uses a bidirectional architecture to encode text and can provide answers to questions based on the context provided."
var_out= obj_model(question= var_ques,context=var_para)
print("question:",var_ques)
print("answer:",var_out['answer'])
