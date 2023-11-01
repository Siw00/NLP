import nltk
from nltk.tokenize import word_tokenize

choice = input("Enter choice exp: ")

if choice == '2':
    # 2 In Google Colab, develop a Simple Python Program to count the number of words in a paragraph, and the user should be prompted to enter a paragraph through a file.
    # Note: The paragraph should not be hand-typed.
    # AIM: To develop a simple Python program to count the number of words in a paragraph
    from google.colab import files

    VAR_input_file = input("Enter the path to the file of x.txt: ")
    uploaded = files.upload()
    VAR_input_file

    VAR_Count = 0
    with open(VAR_input_file, 'r') as j:
        for VAR_sent in j:
            VAR_tokn = VAR_sent.split()
            VAR_Count += len(VAR_tokn)
        print(VAR_Count)

elif choice == '3a':
    VAR_input = "This is Natural Language Processing"
    VAR_Y = VAR_input.split()[::-1]
    VAR_Y
    VAR_len = []
    for x1 in VAR_Y:
        VAR_len.append(len(x1))
    VAR_res = {}
    for i, j in zip(VAR_Y, VAR_len):
        VAR_res[i] = j
    VAR_res
    import collections

    VAR_S = collections.OrderedDict(sorted(VAR_res.items(), key=lambda x: x[1], reverse=True))
    VAR_S
    VAR_words_maxlength = VAR_S.keys()

    print("Words in decreasing order of length are:", VAR_words_maxlength)

elif choice == '4a':
    nltk.download("punkt")
    nltk.download("stopwords")
    import re
    import pandas as pd
    import matplotlib.pyplot as plt
    import collections

    var_input = """transformers consists of encoder and decoder machine learning is the field of study that gives computers the capability to learn
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
    
    stopwords = nltk.corpus.stopwords.words('english')
    print('The stopwords are:', stopwords)
    var_input = word_tokenize(var_input)
    var1 = [i for i in var_input if i.lower() not in stopwords]
    var_tokenized = ' '.join(var1)
    print(var_tokenized)

else:
    print("Wrong choice")
