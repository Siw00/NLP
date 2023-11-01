print("AIM: To develop a simple python program to count the number of words in a paragraph")

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
