#Reading Books - Book1 - Peter Pan, Book2 - Pride and Prejudice
file = open(r"PPan.txt",encoding="utf8")
Book1 = file.read()
B_1=Book1

file = open(r"Pride.txt",encoding="utf8")
Book2 = file.read()
B_2=Book2

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

##ROUND 2 ------------------------->

# Seprating nouns and verbs from lemmatized words=====>
#BOOK 1
from nltk.corpus import wordnet as wn
nouns = []
verbs = []
for word in lemmatized_Book1:
    for synset in wn.synsets(word):
        if "noun" in synset.lexname() and word not in nouns:
            nouns.append(word)
        elif "verb" in synset.lexname() and word not in verbs:
            verbs.append(word)

print(nouns[:100])
print(verbs[:100])

# Nouns
noun_dic = {}
lt = []
for noun in nouns:
    l = []
    for synset in wn.synsets(noun):
        if "noun" in synset.lexname():
            if synset.lexname()[5:] not in l:
                l.append(synset.lexname()[5:])
    noun_dic[noun] = l

j = 0
for i in noun_dic:
    print(i,":", noun_dic[i])
    j +=1
    if j == 50:
        break

# Verbs
verb_dic = {}
for verb in verbs:
    l = []
    for synset in wn.synsets(verb):
        if "verb" in synset.lexname():
            if synset.lexname()[5:] not in l:
                l.append(synset.lexname()[5:])
    verb_dic[verb] = l
j = 0
for i in verb_dic:
    print(i,":", verb_dic[i])
    j +=1
    if j == 50:
        break

# Nouns
noun_cate_dic = {}
for i in noun_dic:
    for j in noun_dic[i]:
        if j not in noun_cate_dic:
            noun_cate_dic[j] = 1
        else:
            noun_cate_dic[j] +=1
print(noun_cate_dic)

# Verbs
verb_cate_dic = {}
for i in verb_dic:
    for j in verb_dic[i]:
        if j not in verb_cate_dic:
            verb_cate_dic[j] = 1
        else:
            verb_cate_dic[j] +=1
print(verb_cate_dic)

#BOOK2
nouns2 = []
verbs2 = []
for word in lemmatized_Book2:
    for synset in wn.synsets(word):
        if "noun" in synset.lexname() and word not in nouns2:
            nouns2.append(word)
        elif "verb" in synset.lexname() and word not in verbs2:
            verbs2.append(word)

print(nouns2[:100])
print(verbs2[:100])

noun_dic2 = {}
for noun in nouns2:
    d = []
    for synset in wn.synsets(noun):
        if "noun" in synset.lexname():
            if synset.lexname()[5:] not in d:
                d.append(synset.lexname()[5:])
    noun_dic2[noun] = d

j = 0
for i in noun_dic2:
    print(i,":", noun_dic2[i])
    j +=1
    if j == 50:
        break

verb_dic2 = {}
for verb in verbs2:
    d = []
    for synset in wn.synsets(verb):
        if "verb" in synset.lexname():
            if synset.lexname()[5:] not in d:
                    d.append(synset.lexname()[5:])
    verb_dic2[verb] = d
j = 0
for i in verb_dic2:
    print(i,":", verb_dic2[i])
    j +=1
    if j == 50:
        break

noun_cate_dic2 = {}
for i in noun_dic2:
    for j in noun_dic2[i]:
        if j not in noun_cate_dic2:
            noun_cate_dic2[j] = 1
        else:
            noun_cate_dic2[j] +=1
print(noun_cate_dic2)

verb_cate_dic2 = {}
for i in verb_dic2:
    for j in verb_dic2[i]:
        if j not in verb_cate_dic2:
            verb_cate_dic2[j] = 1
        else:
            verb_cate_dic2[j] +=1
print(verb_cate_dic2)



#Representation of Nouns and Verbs ===============================================>
# Bar graph of Noun Category and Frequency in Book1
plt.figure()
plt.bar(noun_cate_dic.keys(),noun_cate_dic.values())
plt.title('Noun Category Vs Frequency(for Book 1)')
plt.xlabel('Noun Category')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
plt.show()

# Bar graph of Verb Category and Frequency in Book1
plt.figure()
plt.bar(verb_cate_dic.keys(),verb_cate_dic.values())
plt.title('Verb Category Vs Frequency(for Book 1)')
plt.xlabel('Verb Category')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
plt.show()


# Bar graph of Noun Category and Frequency in Book2
plt.figure()
plt.bar(noun_cate_dic2.keys(),noun_cate_dic2.values())
plt.title('Noun Category Vs Frequency(for Book 2)')
plt.xlabel('Noun Category')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
plt.show()

# Bar graph of Verb Category and Frequency in Book2
plt.figure()
plt.bar(verb_cate_dic2.keys(),verb_cate_dic2.values())
plt.title('Verb Category Vs Frequency(for Book 2)')
plt.xlabel('Verb Category')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
plt.show()


#=========================================================================================================================

# Entity recongition and Classifiaction=================================>

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk import ne_chunk
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score



tagged_Book2 = nltk.word_tokenize(B_2)
tagged_Book2 = nltk.pos_tag(tagged_Book2)
print(tagged_Book2[:100])

results = ne_chunk(tagged_Book2)
print(results[:100])

pattern = 'NP: {<DT>?<JJ>*<NN>}'
cp = nltk.RegexpParser(pattern)
cs = cp.parse(tagged_Book2)
print(cs[:100])

from nltk.chunk import tree2conlltags
from pprint import pprint
iob_tagged = tree2conlltags(cs)
pprint(iob_tagged[:50])

j = 0
for word, pos, ner in iob_tagged:
    print(word, pos, ner)
    j +=1
    if j == 50:
        break

B_2= nlp(B_2)

for X in B_2[6350:6360]:
    print(X, X.ent_iob_, X.ent_type_)

labels = [x.label_ for x in B_2.ents]

Counter(labels)

tagged_S2 = nltk.word_tokenize(B_1)
tagged_S2 = nltk.pos_tag(tagged_S2)
print(tagged_S2[:100])

results2 = ne_chunk(tagged_S2)
print(results2[:100])

pattern = 'NP: {<DT>?<JJ>*<NN>}'
cp2 = nltk.RegexpParser(pattern)
cs2 = cp2.parse(tagged_S2)
print(cs2[:100])

from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint
iob_tagged2 = tree2conlltags(cs2)
pprint(iob_tagged2[:50])

j = 0
for word, pos, ner in iob_tagged2:
    print(word, pos, ner)
    j +=1
    if j == 10:
        break

S2 = nlp(B_1)

for X in S2[6350:6360]:
    print(X, X.ent_iob_, X.ent_type_)

labels2 = [x.label_ for x in S2.ents]

Counter(labels2)

#-----------------------------------------------------------------------------------

#Performance Measures----------------->

#BOOK1
print("entity_predictions")
entity_pred = []
for X in B_2[6350:6750]:
    if X.ent_type_ == "GPE" or X.ent_type_ == "PERSON" or X.ent_type_ == "ORG"or X.ent_type_ == "FAC" or X.ent_type_ == "LOC":
        entity_pred.append('B-'+X.ent_type_)
    elif X.ent_type_=="":
        entity_pred.append('O')
print(entity_pred)

entity_pred = [['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'B-PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],['B-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['B-PERSON', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],['B-PERSON', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['B-PERSON', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['B-PERSON', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O'], ['B-PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O'], ['B-FAC', 'I-FAC', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['B-PERSON', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O'], ['B-PERSON', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O'], ['B-PERSON', 'O', 'O'], ['B-PERSON', 'O', 'O', 'O','O', 'O', 'O'], ['B-PERSON', 'O', 'O', 'O'], ['B-PERSON', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['B-PERSON', 'I-PERSON','O', 'O'], ['B-PERSON', 'O', 'O'], ['B-PERSON', 'I-PERSON', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O'], ['B-PERSON', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O']]
entity_true = [['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'B-PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],['B-PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['B-PERSON', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],['B-PERSON','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['B-PERSON', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['B-PERSON', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O'], ['B-PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O'], ['B-LOC', 'I-LOC', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['B-PERSON', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O'], ['B-PERSON', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O'], ['B-PERSON', 'O', 'O'], ['B-PERSON', 'O', 'O','O', 'O', 'O', 'O'], ['B-PERSON', 'O', 'O', 'O'], ['B-PERSON', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['B-PERSON','I-PERSON', 'O', 'O'], ['B-PERSON', 'O', 'O'], ['B-PERSON', 'I-PERSON', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['B-PERSON', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O']]
f1_score(entity_true, entity_pred)
accuracy_score(entity_true, entity_pred)

#BOOK2
entity_pred2 = []
for X in S2[7050:7500]:
    if X.ent_type_ == "GPE" or X.ent_type_ == "PERSON" or X.ent_type_ == "ORG" or X.ent_type_ == "FAC" or X.ent_type_ == "LOC":
        entity_pred2.append(X.ent_type_)
    elif X.ent_type_=="":
        entity_pred2.append('O')
print(entity_pred2)

entity_pred2 =['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'PERSON', 'PERSON', 'PERSON', 'PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'ORG', 'ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'GPE', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'PERSON', 'PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'GPE', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'ORG', 'O', 'O', 'O', 'O', 'O', 'PERSON', 'O', 'O', 'O', 'O', 'PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'GPE', 'GPE', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
entity_pred2 = [['O', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['B-PERSON', 'I-PERSON', 'O','O'], ['B-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O'], ['B-PERSON', 'I-PERSON', 'I-PERSON', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O'], ['B-LOC', 'O', 'O'], ['B-ORG', 'I-ORG','I-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O'], ['B-PERSON', 'I-PERSON', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],['B-GPE', 'O', 'O', 'O', 'O'], ['B-PERSON', 'I-PERSON', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['B-LOC', 'O', 'O'], ['B-PERSON','I-PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O'], ['B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],['B-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O']]
entity_true2 = [['O', 'O', 'O', 'O', 'B-PERSON', 'I-PERSON', 'I-PERSON','I-PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['O', 'O','O', 'O'], ['B-PERSON', 'I-PERSON', 'I-PERSON', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O'], ['B-PERSON', 'I-PERSON', 'I-PERSON', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['B-LOC', 'O', 'O'],['B-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['B-PERSON','I-PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O'], ['B-GPE', 'O', 'O', 'O', 'O'], ['B-PERSON','I-PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],['B-LOC', 'O', 'O'], ['B-PERSON', 'I-PERSON', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['B-PERSON', 'I-PERSON','I-PERSON', 'I-PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['B-PERSON', 'I-PERSON','I-PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O']]
f1_score(entity_true2, entity_pred2)
accuracy_score(entity_true2, entity_pred2)


#-------------------------------------------------------------------------------------------------------------------

#Extracting Relationship between Entities

TEXTS = [
"""Mrs. Darling loved to have everything just so, and Mr. Darling had a
passion for being exactly like his neighbours; so, of course, they had
a nurse. As they were poor, owing to the amount of milk the children
drank, this nurse was a prim Newfoundland dog, called Nana, who had
belonged to no one in particular until the Darlings engaged her. She had
always thought children important, however, and the Darlings had become
acquainted with her in Kensington Gardens, where she spent most of her
spare time peeping into perambulators, and was much hated by careless
nursemaids, whom she followed to their homes and complained of to their
mistresses. She proved to be quite a treasure of a nurse. How thorough
she was at bath-time, and up at any moment of the night if one of her
charges made the slightest cry. Of course her kennel was in the nursery.
She had a genius for knowing when a cough is a thing to have no patience
with and when it needs stocking around your throat. She believed to her
last day in old-fashioned remedies like rhubarb leaf, and made sounds of
contempt over all this new-fangled talk about germs, and so on. It was a
lesson in propriety to see her escorting the children to school, walking
sedately by their side when they were well behaved, and butting them
back into line if they strayed. On John's footer [in England soccer
was called football, “footer” for short] days she never once forgot his
sweater, and she usually carried an umbrella in her mouth in case of
rain. There is a room in the basement of Miss Fulsom's school where the
nurses wait. They sat on forms, while Nana lay on the floor, but that
was the only difference. They affected to ignore her as of an inferior
social status to themselves, and she despised their light talk. She
resented visits to the nursery from Mrs. Darling's friends, but if they
did come she first whipped off Michael's pinafore and put him into the
one with blue braiding, and smoothed out Wendy and made a dash at John's
hair."""
]

ner_model = nlp
def ner_text():
    doc = ner_model(TEXTS[0])
    for entity in doc.ents:
        print(entity.label_,' ',entity.text)
ner_text()

def filter_spans(spans):
    get_sort_key = lambda span: (span.end - span.start, -span.start)
    sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
    result = []
    seen_tokens = set()
    for span in sorted_spans:
        if span.start not in seen_tokens and span.end - 1 not in seen_tokens:
            result.append(span)
        seen_tokens.update(range(span.start, span.end))
    result = sorted(result, key=lambda span: span.start)
    return result

def main(ner_model):
    nlp = ner_model
    print("Processing %d texts" % len(TEXTS))
    for text in TEXTS:
        doc = nlp(text)
        relations = extract_per_relations(doc)
        for r1, r2 in relations:
            print("{:<10}\t{}\t{}".format(r1.text, r2.ent_type_, r2.text))

def extract_per_relations(doc):
    spans = list(doc.ents) + list(doc.noun_chunks)
    spans = filter_spans(spans)
    with doc.retokenize() as retokenizer:
        for span in spans:
            retokenizer.merge(span)
    relations = []
    for per in filter(lambda w: w.ent_type_ == "PERSON", doc):
        if per.dep_ in ("attr", "dobj"):
            subject = [w for w in per.head.lefts if w.dep_ == "nsubj"]
            if subject:
                subject = subject[0]
                relations.append((subject, per))
        elif per.dep_ == "pobj" and per.head.dep_ == "prep":
            relations.append((per.head.head, per))
    return relations

main(ner_model)

TEXTS = ["""The evening altogether passed off pleasantly to the whole family.
      Mrs. Bennet had seen her eldest daughter much admired by the
      Netherfield party. Mr. Bingley had danced with her twice, and she
      had been distinguished by his sisters. Jane was as much gratified
      by this as her mother could be, though in a quieter way.
      Elizabeth felt Jane’s pleasure. Mary had heard herself mentioned
      to Miss Bingley as the most accomplished girl in the
      neighbourhood; and Catherine and Lydia had been fortunate enough
      never to be without partners, which was all that they had yet
      learnt to care for at a ball. They returned, therefore, in good
      spirits to Longbourn, the village where they lived, and of which
      they were the principal inhabitants. They found Mr. Bennet still
      up. With a book he was regardless of time; and on the present
      occasion he had a good deal of curiosity as to the event of an
      evening which had raised such splendid expectations. He had
      rather hoped that his wife’s views on the stranger would be
      disappointed; but he soon found out that he had a different story
      to hear."""
]

ner_text()

main(ner_model)