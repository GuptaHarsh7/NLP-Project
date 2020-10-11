#Reading Books - Book1 - Peter Pan, Book2 - Pride and Prejudice
file = open(r"C:\Users\Lenovo\Desktop\POS_tagger\PPan.txt",encoding="utf8")
Book1 = file.read()

file = open(r"C:\Users\Lenovo\Desktop\POS_tagger\Pride.txt",encoding="utf8")
Book2 = file.read()

# DATA PREPROCESSING STEP ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

#Step 1 - Python Punctuations ------------------------------------------
import string
print("String Punctuations:- " + string.punctuation)

translate = str.maketrans("","",string.punctuation) 
# basically, translating punctuations in the Books to something suotable for us or removing them
Book1=Book1.translate(translate)
Book2=Book2.translate(translate)

#Step 2 - Removing things we don't want from the books-----------------------------------
import re

# Removing acknowledgement (if any)
Book1 = re.sub("The[\s\S]*CONTENTS","",Book1)
# Removing the Transciber's Notes (if any)
Book1 = re.sub(r"Transcriber’s[\s\S]*",r"",Book1)

print(Book1[:1000])


Book2 = re.sub("The[\s\S]*CONTENTS","",Book2)
Book2 = re.sub(r"Transcriber’s[\s\S]*",r"",Book2)
print(Book2[:1000])

# Removing Chapter Names
Book1=re.sub("[A-Z]{2,}","",Book1)
print(Book1[:1000])

Book2=re.sub("[A-Z]{2,}","",Book2)
print(Book2[:1000])


#Lowercasing text in books
Book1 = Book1.lower()
Book2=Book2.lower()

# Removing Chapter Numbers
Book1=re.sub("[0-9]+","",Book1)
print(Book1[:1000])

Book2=re.sub("[0-9]+","",Book2)
print(Book2[:1000])


#STEP 3 ---------------------------------------------------------------------------------------

# We do this because first when we split the words we create a list of the words present in the book after that
# we join them together and through this we make sure to remove line ends and extra spaces etc.. basically things not 
# recquired by us. So, in the end we get a compiled text that we can tokenize and do our POS on.
Book1=Book1.split()
Book2=Book2.split()

Book1 = " ".join(Book1)
Book2 = " ".join(Book2)
wordcloudBook1=Book1
wordcloudBook2=Book2
print(Book1[:1000])
print(Book2[:1000])


#STEP 4 - Tokenization of the compiled texts ---------------------------------------------------
import nltk
from nltk.tokenize import word_tokenize

Book1 = word_tokenize(Book1)
print(Book1[:1000])

Book2 = word_tokenize(Book2)
print(Book2[:1000])


#STEP 5 - Lemmataization - converting the words to their base forms----------------------------
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Lemmatisation
Lemma = WordNetLemmatizer()
# creating a set of stopwords 
stop_words = set(stopwords.words('english'))
# creating an empty list meant for lemmatizing
lemmatized_Book1=[]
# traversing all the words in the book
for word in Book1:
# if the word has a length greater than or equal to 2 and is not a stopword
    if len(word) >= 2 and word not in stop_words:
    # then we append the word into the list lemmatized_T after performing lemmatization
    # using the lemmatize() function
        lemmatized_Book1.append(Lemma.lemmatize(word))
# printing the list of lemmatized words
print(lemmatized_Book1[:1000])

#Similarly for Book2
lemmatized_Book2=[]
for word in Book2:
    if len(word) >= 2 and word not in stop_words:
        lemmatized_Book2.append(Lemma.lemmatize(word))
print(lemmatized_Book2[:1000])


#REPRESENTATION STEP |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

# STEP 1 -  Evaluating frequency distribution of tokens -----------------------------------------------------------------
# The nltk library function FreqDist() returns a dictionary containing key␣→value pairs where values are the frequency of
# the keys. Here keys are the Tokens.

#### for Book1
freq_dist = nltk.FreqDist(lemmatized_Book1)
# create a dictionary
dict_word = {}
# find the k words occuring f number of times and store in the dictionary
for i in freq_dist.keys():
    if freq_dist[i] not in dict_word:
        dict_word[freq_dist[i]] = 1
    else:
        dict_word[freq_dist[i]] += 1
# plotting a scatter plot diagram of the frequency distribution and tokens

import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plt.scatter(dict_word.keys(),dict_word.values())
plt.xlabel("No. of Tokens")
plt.ylabel("Frequency")
plt.title("Frequency Distribution of tokens for Book1")
plt.show()

#Similarly for Book2
freq_dist = nltk.FreqDist(lemmatized_Book2)
dict_word = {}
for i in freq_dist.keys():
    if freq_dist[i] not in dict_word:
        dict_word[freq_dist[i]] = 1
    else:
        dict_word[freq_dist[i]] += 1

import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plt.scatter(dict_word.keys(),dict_word.values())
plt.xlabel("No. of Tokens")
plt.ylabel("Frequency")
plt.title("Frequency Distribution of tokens for Book2")
plt.show()

# STEP 2 - WORD CLOUD -----------------------------------------------------------------------------------------

from wordcloud import WordCloud, STOPWORDS
wordcloud = WordCloud(width = 800, height=800,
background_color='white',
min_font_size=10).generate(wordcloudBook1)
plt.figure(figsize=(8,8),facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Book1 Word Cloud")
plt.show()

#Similarly for Book2
from wordcloud import WordCloud, STOPWORDS
wordcloud = WordCloud(width = 800, height=800,
background_color='white',
min_font_size=10).generate(wordcloudBook2)
plt.figure(figsize=(8,8),facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Book2 Word Cloud")
plt.show()

#STEP 3 - Removing StopWords and representing Word Cloud ------------------------------------------------------

# WordCloud after removing Stopwords
wordcloud = WordCloud(width = 800, height=800,
background_color='white',
stopwords = set(STOPWORDS),
min_font_size=10).generate(wordcloudBook1)
plt.figure(figsize=(8,8),facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

# For Book2
wordcloud = WordCloud(width = 800, height=800,
background_color='white',
stopwords = set(STOPWORDS),
min_font_size=10).generate(wordcloudBook2)
plt.figure(figsize=(8,8),facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

# Step 4 - Frequency Distribution for word length --------------------------------------------------------------

len_list={}
for i in range(len(lemmatized_Book1)):
    if len(lemmatized_Book1[i]) not in len_list:
        len_list[len(lemmatized_Book1[i])] = 1
    else:
        len_list[len(lemmatized_Book1[i])] += 1
keys = list(len_list.keys())
values = list(len_list.values())
plt.figure(figsize=(10,10))
plt.bar(keys,values)
plt.xlabel("Word Length")
plt.ylabel("Frequency")
plt.title("Frequency Distribution of word length for Book1")
plt.show()

#For Book2
len_list={}
for i in range(len(lemmatized_Book2)):
    if len(lemmatized_Book2[i]) not in len_list:
        len_list[len(lemmatized_Book2[i])] = 1
    else:
        len_list[len(lemmatized_Book2[i])] += 1
keys = list(len_list.keys())
values = list(len_list.values())
plt.figure(figsize=(10,10))
plt.bar(keys,values)
plt.xlabel("Word Length")
plt.ylabel("Frequency")
plt.title("Frequency Distribution of word length fot Book2")
plt.show()


# Part Of Speech Tagging Process |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||


#STEP 1 - Incorporating Corpus and training on it -----------------------------------------------------------------------
from nltk.corpus import brown#Using brown corpus
brown_tagged_sents = brown.tagged_sents(categories=['fiction','romance','adventure','mystery','humor','science_fiction'])
brown_sents = brown.sents(categories=['fiction','romance','adventure','mystery','humor','science_fiction'])
#These categories looked best in relation to the books chose for the project

#Random data selection for Training and Testing
size = int(len(brown_tagged_sents) * 0.9)

train_sents = brown_tagged_sents[:size]
test_sents = brown_tagged_sents[size:]

train_sents[:1000]
test_sents[:1000]
#Without random shuffling cause using bigram model


#Using backoff method and Bigram Model for POS Tag Training
tag0 = nltk.DefaultTagger('NN')#NN because it's better to  give a Noun tag as most of the unknown words are names.
tag1 = nltk.UnigramTagger(train_sents, backoff=tag0)
tag2 = nltk.BigramTagger(train_sents, backoff=tag1)
print(tag2.evaluate(test_sents))


#Using the model on our books to get tags
Tagged_Book1 =tag2.tag(lemmatized_Book1)

f = open("Tagged_Book1.txt","a")
f.write(str(Tagged_Book1))
f.close()

Tagged_Book2=tag2.tag(lemmatized_Book2)

f = open("Tagged_Book2.txt","a")
f.write(str(Tagged_Book2))
f.close()

Freq_dist_tag1 = nltk.FreqDist([t for (w, t) in Tagged_Book1])
Freq_dist_tag2 = nltk.FreqDist([t for (w, t) in Tagged_Book2])

#Frequency Distribution of TAGS -----------------------------------------------------------------------------------

#Keys in Corpus
print("No. of tags", len(Freq_dist_tag1.keys()))
Freq_dist_tag1.keys()

keys = (list(Freq_dist_tag1.keys()))
# creating a list of the frequency of the various tags
for i in keys:
    if '$' in i:
        keys[keys.index(i)] = keys[keys.index(i)].strip('$') + '1'
    elif '$$' in i:
        keys[keys.index(i)] = keys[keys.index(i)].strip('$$') + '1'
#Removing $ from our words beacuse that causes problem instead using 1
values = list(Freq_dist_tag1.values())
# plotting a bar plot diagram of the frequency distribution
plt.figure(figsize=(10,10))
plt.bar(keys,values)
plt.xlabel("Tags")
plt.xticks(rotation=45)
plt.ylabel("Frequency")
plt.title("Frequency Distribution of tags in Book1")
plt.show()

#Similarly for Book2
keys = (list(Freq_dist_tag2.keys()))

for i in keys:
    if '$' in i:
        keys[keys.index(i)] = keys[keys.index(i)].strip('$') + '1'
    elif '$$' in i:
        keys[keys.index(i)] = keys[keys.index(i)].strip('$$') + '1'
values = list(Freq_dist_tag2.values())

plt.figure(figsize=(10,10))
plt.bar(keys,values)
plt.xlabel("Tags")
plt.xticks(rotation=45)
plt.ylabel("Frequency")
plt.title("Frequency Distribution of tags in Book2")
plt.show()

#===========================================================================================================================