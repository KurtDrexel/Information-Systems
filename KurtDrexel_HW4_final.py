#by Kurt Drexel, April 2024
#call this program using: python KurtDrexel_HW4_final.py inputdirectoryname outputdirectoryname
#after completing the corpus the program will then prompt the user for the word to search for
#input this term and the program will return a list of files the word occurs in
#Origional code by Nia Klender and Kurt Drexel, February 2024
import nltk
from nltk.tokenize import RegexpTokenizer
import os
from collections import OrderedDict
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import math
import copy
from collections import defaultdict
#downloads needed to get NLTK to work correctly:
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')

fullstart = time.perf_counter() #for measuring the time the program takes to run

#stop word list
stopwords = ["a","about","above","according","across","actually","adj","after","afterwards","again","against","all",
             "almost","alone","along","already","also","although","always","among","amongst","an","and","another","any",
             "anybody","anyhow","anyone","anything","anywhere","are","area","areas","aren't","around","as","ask","asked",
             "asking","asks","at","away","b","back","backed","backing","backs","be","became","because","become","becomes",
             "becoming","been","before","beforehand","began","begin","beginning","behind","being","beings","below","beside",
             "besides","best","better","between","beyond","big","billion","both","but","by","c","came","can","can't",
             "cannot","caption","case","cases","certain","certainly","clear","clearly","co","come","could","couldn't",
             "d","did","didn't","differ","different","differently","do","does","doesn't","don't","done","down","downed",
             "downing","downs","during","e","each","early","eg","eight","eighty","either","else","elsewhere","end","ended",
             "ending","ends","enough","etc","even","evenly","ever","every","everybody","everyone","everything","everywhere",
             "except","f","face","faces","fact","facts","far","felt","few","fifty","find","finds","first","five","for",
             "former","formerly","forty","found","four","from","further","furthered","furthering","furthers","g","gave",
             "general","generally","get","gets","give","given","gives","go","going","good","goods","got","great","greater",
             "greatest","group","grouped","grouping","groups","h","had","has","hasn't","have","haven't","having","he",
             "he'd","he'll","he's","hence","her","here","here's","hereafter","hereby","herein","hereupon","hers","herself",
             "high","higher","highest","him","himself","his","how","however","hundred","i","i'd","i'll","i'm","i've","ie",
             "if","important","in","inc","indeed","instead","interest","interested","interesting","interests","into","is",
             "isn't","it","it's","its","itself","j","just","k","l","large","largely","last","later","latest","latter",
             "latterly","least","less","let","let's","lets","like","likely","long","longer","longest","ltd","m","made",
             "make","makes","making","man","many","may","maybe","me","meantime","meanwhile","member","members","men",
             "might","million","miss","more","moreover","most","mostly","mr","mrs","much","must","my","myself","n","namely",
             "necessary","need","needed","needing","needs","neither","never","nevertheless","new","newer","newest","next",
             "nine","ninety","no","nobody","non","none","nonetheless","noone","nor","not","nothing","now","nowhere",
             "number","numbers","o","of","off","often","old","older","oldest","on","once","one","one's","only","onto",
             "open","opened","opens","or","order","ordered","ordering","orders","other","others","otherwise","our","ours",
             "ourselves","out","over","overall","own","p","part","parted","parting","parts","per","perhaps","place",
             "places","point","pointed","pointing","points","possible","present","presented","presenting","presents",
             "problem","problems","put","puts","q","quite","r","rather","really","recent","recently","right","room","rooms",
             "s","said","same","saw","say","says","second","seconds","see","seem","seemed","seeming","seems","seven","seventy",
             "several","she","she'd","she'll","she's","should","shouldn't","show","showed","showing","shows","sides","since",
             "six","sixty","small","smaller","smallest","so","some","somebody","somehow","someone","something","sometime",
             "sometimes","somewhere","state","states","still","stop","such","sure","t","take","taken","taking","ten","than",
             "that","that'll","that's","that've","the","their","them","themselves","then","thence","there","there'd",
             "there'll","there're","there's","there've","thereafter","thereby","therefore","therein","thereupon","these",
             "they","they'd","they'll","they're","they've","thing","things","think","thinks","thirty","this","those","though",
             "thought","thoughts","thousand","three","through","throughout","thru","thus","to","today","together","too",
             "took","toward","towards","trillion","turn","turned","turning","turns","twenty","two","u","under","unless",
             "unlike","unlikely","until","up","upon","us","use","used","uses","using","v","very","via","w","want","wanted",
             "wanting","wants","was","wasn't","way","ways","we","we'd","we'll","we're","we've","well","wells","were",
             "weren't","what","what'll","what's","what've","whatever","when","whence","whenever","where","where's","whereafter","whereas","whereby","wherein","whereupon","wherever","whether","which","while","whither","who","who'd","who'll","who's","whoever","whole","whom","whomever","whose","why","will","with","within","without","won't","work","worked","working","works","would","wouldn't","x","y","year","years","yes","yet","you","you'd","you'll","you're","you've","young","younger","youngest","your","yours","yourself","yourselves","z"]
#This stopword list could have been iterated through by the file but i decided to transfer it to an array.
#I decided to do it this way because it would keep with the way me and my partner had done it during hw1.
#for simplicity allowing me to simply paste the new stoplist inside the program and run it

tokens = {}#dictionary to store all tokens and the number of times they occur
#Occurances = defaultdict(lambda: defaultdict(int))
Occurances = {}#defaultdict(lambda: defaultdict(int))
#defaultdict will help to account for new words detected in dict adding them and setting the counts accordingly 
ProcessedTok = {}#hold processed tokens
Titles =[]#list of filenames without html
Weights = {}#dictionary of tf*idf weights
TermOccurances = defaultdict(lambda: defaultdict(int))
#two forms of input one for notebooks the other for normal python programs
directory_path = sys.argv[1] #takes command line argument for the directory where the input files are located
#directory_path = input() #used to get input in jupyter
output_directory = sys.argv[2] #takes command line argument for the output directory (does not need to exist already)
#output_directory = input() #used to get input in jupyter

if not os.path.exists(output_directory):#if output directory does not exist, create it
    os.makedirs(output_directory)
words = [] #array to hold words gotten from each document
for filename in os.listdir(directory_path):#iterate through files
    starttime = time.perf_counter()
    DocNum = filename.rsplit('.')[0]  # get begining of filename without html
    Titles.append(DocNum)
    words.clear()#clear word array to start a new document
    f = os.path.join(directory_path, filename)
    if os.path.isfile(f): #check if it's a file
        with open(f, 'r', encoding = 'latin-1') as f:#open file to read
            for line in f: #for each line in the file
                #tokenizer = RegexpTokenizer(r'\w+')
                tokenizer = RegexpTokenizer(r'<|>|\'|\w+') #remove punctuation from tokens, except < and >
                                                        #these will be used to identify and remove html syntax
                newtokens = tokenizer.tokenize(line) #tokenize line
                words.extend(newtokens) #add tokenized words to the word array
            flag = 0
            discard_pile = []
            for i in range(len(words)):
                words[i] = words[i].lower()
                if words[i] == "<": #detect if it is part of html syntax
                    flag += 1
                    discard_pile.insert(0,i)
                elif words[i] == ">": #detect end of html syntax
                    flag -= 1
                    discard_pile.insert(0,i)
                elif flag > 0: #if flag is up discard current tokenized word
                    discard_pile.insert(0, i)
                elif len(words[i]) == 1:#words of length 1 are stopped by stoplist but this will prevent single digits as well
                    discard_pile.insert(0,i)
            for i in discard_pile: #iterate through discard pile and remove them all from the list of words
                words.pop(i) #ordered by indices in reverse, so it doesn't get mixed up when removing

            for stopword in stopwords: #iterate through stopwords
                while stopword in words:
                    words.remove(stopword) #stopword found in words array, remove stopword
        #print(len(words))
        ProcessedTok[DocNum] = copy.deepcopy(words)
        #this needs to be a deepcopy as a normal copy will become useless after words is cleared when
        #iterating to the next file
        for token in words:#iterate through words
            if token in tokens:
                tokens[token] += 1#if token occurs more then once increment count
            else:
                tokens[token] = 1#first occurance of token
        for token in words:
            if not token in Occurances:
                Occurances[token] = []
                Occurances[token].append(DocNum)
            else:
                if not DocNum in Occurances[token]:
                    Occurances[token].append(DocNum)
                    #TermOccurances[token][DocNum] += 1#termoccurances will hold the term and each document it appears in

#delete un-neccessary tokens
keys = list(tokens.keys())
for i in range(len(keys)):#iterate through words
    if tokens[keys[i]] == 1:#if word occurs once
        del tokens[keys[i]]  # delete once occuring word
        Pos = Occurances[keys[i]][0]  # position of deleted word
        ProcessedTok[Pos].remove(keys[i])  # remove from dictonary
        del Occurances[keys[i]]  # delete in occurances
        i -= 1

#tf*idf and normalization
Sum =0#initialize to 0
for DocNum in Titles: #iterate through document titles
    #print(DocNum)
    Sum = 0#reset sum for each new document
    Weights[DocNum] = {}
    for token in Occurances:  # iterate through tokens in document
        weight = 0#intialize new weight for each term in document
        if DocNum in Occurances[token]:#iterate through tokenized words
            df = len(Occurances[token])  # of documents the token appears in or document frequency (tf)####always =1
            tf = ProcessedTok[DocNum].count(token)  # of times t occurs in doc d or term frequency (idf)
            weight = math.log(1 + tf) * math.log10(503 / df)#calculate TF*IDF
            Sum += weight*weight #square w
            Weights[DocNum][token] = weight  # set token frequency
    sqrt_sum = math.sqrt(Sum)
    for token in Weights[DocNum]:
        Weights[DocNum][token] /=sqrt_sum#calculate new frequency
#Weights = {docnum1={token1=2,token2=5}, docnum2={token1=9,token2=4}}

Occurances = OrderedDict(sorted(Occurances.items()))#order the current dictionary #alphabetical

#create dictionary and postings files
with open(os.path.join(output_directory, 'dictionary.txt'), 'w') as dict_file:
    with open(os.path.join(output_directory, 'postings.txt'), 'w') as postings_file:
        curr_line = 1
        # Iterate through terms and write entries to dictionary and postings
        for term, document in Occurances.items():#iterate through terms
            num_docs_containing_term = len(document)  # Number of documents containing the term
            # Write postings for the term
            for doc in document:#iterate through documents term occurs in
                #postings_file.write(f"{curr_doc},{doc},{Weights[doc][term]}\n")
                postings_file.write(f"{doc},{Weights[doc][term]}\n")#write to postings file
            dict_file.write(f"{term}\n{num_docs_containing_term}\n{curr_line}\n")#write to dict file
            curr_line = curr_line+num_docs_containing_term #update current line

#After running program using output_directory, iterate through dictionary for all words in input
#example dog cat mouse
#find postings of dog in form of dog \n #of files with term \n line in postings file
#do for all three terms and store in dictionary. If more then 1 term occurs in documnet
#term weight 1 + term weight 2 = term weight to print
#print total dictionary with document numbers and combined weights

search_phrase = input("Enter the phrase you want to search for: ").lower()

# Split the search phrase into individual words
search_words = search_phrase.split()

# Dictionary to store frequencies of each word in each document
word_frequencies = {}

# Iterate through each word in the search phrase
for word in search_words:
    with open(os.path.join(output_directory, 'dictionary.txt'), 'r') as dict_file:
        lines = dict_file.readlines()  # Read all lines from the file

        # Iterate through each line in the file
        for i in range(0, len(lines), 3):
            term = lines[i].strip().lower()
            if term == word:  # Check if the word in the line matches the search word
                num_docs_containing_term = int(lines[i + 1].strip())  #number of documents containing the term
                line_in_postings = int(lines[i + 2].strip())  # position in the postings file

                print(f"Word '{word}' found in the dictionary.")
                #print(f"Number of documents containing the term: {num_docs_containing_term}")
                #print(f"Position in the postings file: {line_in_postings}")

                # Update word frequencies for the current word
                with open(os.path.join(output_directory, 'postings.txt'), 'r') as postings_file:
                    for line_num, line in enumerate(postings_file, 1):
                        if line_num >= line_in_postings:  # Start processing from the specified line
                            doc_id, weight = line.strip().split(',')#split at comma
                            if doc_id not in word_frequencies:#add to dictionary
                                word_frequencies[doc_id] = {}
                            if word not in word_frequencies[doc_id]:
                                word_frequencies[doc_id][word] = 0
                            word_frequencies[doc_id][word] += float(weight)
                            if line_num >= line_in_postings + num_docs_containing_term - 1:
                                break

                break  # Exit the loop since the word has been found
        else:
            print(f"Word '{word}' not found in the dictionary.")

# Combine frequencies for the same document
for doc_id, frequencies in word_frequencies.items():
    combined_frequency = sum(frequencies.values())
    word_frequencies[doc_id] = combined_frequency

#sort dictionary in descending order by frequency
sorted_word_frequencies = dict(sorted(word_frequencies.items(), key=lambda item: item[1], reverse=True))

# Print out results with frequencies for each document in descending order
if sorted_word_frequencies:
    print("\nSearch results:")
    for doc_id, combined_frequency in sorted_word_frequencies.items():
        print(f"Document ID: {doc_id}, Frequency: {combined_frequency}")
else:
    print("No words from the search phrase were found in the dictionary.")
