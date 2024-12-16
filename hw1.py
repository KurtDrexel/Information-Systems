#by Nia Klender and Kurt Drexel, February 2024
#call this program using: python hw1.py inputdirectoryname outputdirectoryname

import nltk
from nltk.tokenize import RegexpTokenizer
import os
from collections import OrderedDict
import sys
import time
#downloads needed to get NLTK to work correctly:
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')

fullstart = time.perf_counter() #for measuring the time the program takes to run

#stop word list, to be constantly updated as see fit
stopwords = ["the", "and", "to", "on", "<", ">", "--", "!", "'", "''", ":", ")", "(", ",", "|", "%", "-", "." ,"[","]",
             "a","A","b","B","c","C","d","D","e","E","f","F","g","G","h","H","i","I","j","J","k","K","l","L","m","M",
             "n","N","o","O","p","P","q","Q","r","R","s","S","t","T","u","U","v","V","w","W","x","X","y","Y","z","Z",
             "*","is","=","==","``","`","?", "so","body=", ";", "{", "}", "$", "#", "&", "@", "of", "the", "for", "by",
             "as", "be", "are", "was", "it", "this", "an"]
            #body= is more of a bandaid, it's a html tag

#dictionary to store all tokens and the number of times they occur
tokens = {}

#list of tuples, contains data about time elapsed for each file
timedata = []
    
#encodings = ['utf-8', 'latin-1', 'iso-8859-1'] #latin-1 will be used if utf-8 cannot be used

directory_path = sys.argv[1] #takes command line argument for the directory where the input files are located
#directory_path = input() #used to get input in jupyter
output_directory = sys.argv[2] #takes command line argument for the output directory (does not need to exist already)
#output_directory = input() #used to get input in jupyter

if not os.path.exists(output_directory):#if output directory does not exist, create it
    os.makedirs(output_directory)

words = [] #array to hold words gotten from each document

for filename in os.listdir(directory_path):#iterate through files
#filename = '062.html'
#for i in range(1):    #used for testing a single file:
    starttime = time.perf_counter()
    prefix = ""
    words.clear()#clear word array to start a new document
    f = os.path.join(directory_path, filename)
    if os.path.isfile(f): #check if it's a file
        with open(f, 'r', encoding = 'latin-1') as f:#open file to read
            for line in f: #for each line in the file
                #tokenizer = RegexpTokenizer(r'\w+')
                tokenizer = RegexpTokenizer(r'<|>|\w+') #remove punctuation from tokens, except < and >
                                                        #these will be used to identify and remove html syntax
                newtokens = tokenizer.tokenize(line) #tokenize line
                words.extend(newtokens) #add tokenized words to the word array

            flag = 0
            discard_pile = []
            for i in range(len(words)):
                if words[i] == "<": #detect if it is part of html syntax
                    flag += 1
                    discard_pile.insert(0,i)
                elif words[i] == ">": #detect end of html syntax
                    flag -= 1
                    discard_pile.insert(0,i)
                elif flag > 0: #if flag is up discard current tokenized word
                    discard_pile.insert(0, i)
            for i in discard_pile: #iterate through discard pile and remove them all from the list of words
                words.pop(i) #ordered by indices in reverse, so it doesn't get mixed up when removing
            #print("after removing html tags:", words)

            for stopword in stopwords: #iterate through stopwords
                while stopword in words:
                    words.remove(stopword) #stopword found in words array, remove stopword
            #print("after removing stopwords:", words)

        prefix = filename.rsplit('.')[0]#get begining of filename without (.html) to be concated for output file
        #print("prefix", prefix)
        outputfile = prefix + '_output.txt'#create outputfilename
        outputfile = os.path.join(output_directory,outputfile)
        #print("opf", outputfile)

        with open(outputfile, 'w') as file:#open output file 
            for token in words:#iterate through tokenized word
                try:
                    file.write(token.lower())#write token to file
                    file.write("\n")#add newline
                    #increment ocurrence count for that token in the tokens dictionary, or add it as a key if not there
                    if token in tokens:
                        tokens[token.lower()] += 1
                    else:
                        tokens[token.lower()] = 1
                except:
                    print("could not print a token in", prefix)
    endtime = time.perf_counter()
    time_elapsed = endtime - starttime
    timedata.append((prefix,time_elapsed))

tokens_alphabetical = OrderedDict(sorted(tokens.items()))#sort words alphabetically
tokens_frequency = dict(sorted(tokens.items(), key = lambda item: item[1], reverse=True))#organize words by frequency and in reverse order

#output full files for alphabetical and frequency sorted tokens
outputfile = os.path.join(output_directory, "tokenlist_alphabetical.txt")  # create output file for alphabetical words

with open(outputfile, 'w') as file:  # write alphabetical output to file
    counter = 0
    for token in tokens_alphabetical:
        file.write(token)
        for i in range(max(40 - len(token), 5)):  # used to keep output file organized by keeping the columns lined up
            file.write(" ")
        file.write("|         ")
        file.write(str(tokens[token]))
        file.write("\n")
        counter += 1

outputfile = os.path.join(output_directory, "tokenlist_frequency.txt")  # create frequency file

with open(outputfile, 'w') as file:  # write to output file
    counter = 0
    for token in tokens_frequency:  # iterate through tokens
        file.write(token)
        for i in range(max(40 - len(token), 5)):  # keep output file organized by keeping columns lined up
            file.write(" ")
        file.write("|         ")
        file.write(str(tokens[token]))
        file.write("\n")
        counter += 1


""" #output only first and last 50 lines of each file
outputfile = os.path.join(output_directory,"tokenlist_alphabetical.txt")#create output file for alphabetical words

with open(outputfile, 'w') as file:#write alphabetical output to file
    counter = 0
    for token in tokens_alphabetical:
        if counter < 50 or counter >= len(tokens_alphabetical.keys()) - 50:#get first 50 and last 50 in alphabetical list
            file.write(token)
            for i in range(max(40 - len(token),5)):#used to keep output file organized by keeping the columns lined up
                file.write(" ")
            file.write("|         ")
            file.write(str(tokens[token]))
            file.write("\n")
        elif counter == 51:
            file.write("\n\n\n")#easy separation of first 50 and last 50
        counter += 1
        
outputfile = os.path.join(output_directory, "tokenlist_frequency.txt")#create frequency file

with open(outputfile, 'w') as file:#write to output file
    counter = 0
    for token in tokens_frequency:#iterate through tokens 
        if counter < 50 or counter >= len(tokens_frequency.keys()) - 50:
            file.write(token)
            for i in range(max(40 - len(token),5)):#keep output file organized by keeping columns lined up
                file.write(" ")
            file.write("|         ")
            file.write(str(tokens[token]))
            file.write("\n")
        elif counter == 51:
            file.write("\n\n\n")#easy separation of first 50 and last 50
        counter += 1
"""

#for recording time data
"""
print("Time data:")
print(timedata)

fullstop = time.perf_counter()
totaltime = fullstop - fullstart
print("Total time in sec:", totaltime, "in minutes:", totaltime/60)
"""