import pandas as pd
import numpy as np
import os
import textblob
from textblob import TextBlob


import wordcloud
from wordcloud import WordCloud

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import contractions
nltk.download('averaged_perceptron_tagger')
keywordSet = {"don't", "never", "nothing", "nowhere", "noone", "none", "not",
              "hasn't", "hadn't", "can't", "couldn't", "shouldn't", "won't",
              "wouldn't", "don't", "doesn't", "didn't", "isn't", "aren't", "ain't"}


def negatif_tokens(current_wordset):
    j = 0
    while j < len(current_wordset) - 1:
        if current_wordset[j] in keywordSet:
            current_wordset[j] = 'not_' + current_wordset[j + 1]
            current_wordset.pop(j + 1)
        j += 1
    return current_wordset


# stop words
import spacy
import gensim


def remove_stopwords(text_tokens):
    text_without_sw = []
    for word in text_tokens:
        # checking word is stopword or not
        if word not in all_stopwords:
            text_without_sw.append(word)

    return text_without_sw


nltk.download('stopwords')
# list of stopwords from nltk
stopwords_nltk = list(stopwords.words('english'))
sp = spacy.load('en_core_web_sm')
# list of stopwords from spacy
stopwords_spacy = list(sp.Defaults.stop_words)
# list of stopwords from gensim
stopwords_gensim = list(gensim.parsing.preprocessing.STOPWORDS)

# unique stopwords from all stopwords
all_stopwords = []
all_stopwords.extend(stopwords_nltk)
all_stopwords.extend(stopwords_spacy)
all_stopwords.extend(stopwords_gensim)
# all unique stop words
all_stopwords = list(set(all_stopwords))

from collections import Counter


def remove_frequent(tokens, number_to_remove):
    counted = Counter(tokens)
    most_occur = counted.most_common(number_to_remove)
    for tupl in most_occur:
        counted.pop(tupl[0])
    return counted.keys()


nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tag import StanfordTagger
from nltk.corpus.reader.wordnet import VERB, NOUN, ADJ, ADV

dict_pos_map = {
    # Look for NN in the POS tag because all nouns begin with NN
    'NN': NOUN,
    # Look for VB in the POS tag because all nouns begin with VB
    'VB': VERB,
    # Look for JJ in the POS tag because all nouns begin with JJ
    'JJ': ADJ,
    # Look for RB in the POS tag because all nouns begin with RB
    'RB': ADV
}
lm = WordNetLemmatizer()


def lemmatize_with_pos_text(tokens):
    normalized_tokens = []
    for tuples in nltk.pos_tag(tokens):
        temp = tuples[0]
        if tuples[1] == "NNP" or tuples[1] == "NNPS":
            continue
        if tuples[1][:2] in dict_pos_map.keys():
            temp = lm.lemmatize(tuples[0].lower(), pos=dict_pos_map[tuples[1][:2]])
        normalized_tokens.append(temp)
    return normalized_tokens


nltk.download('punkt')
import string
from nltk import word_tokenize, punkt


def tokenize_text(text):
    no_punct_seq_tokens = [token for token in word_tokenize(text) if token not in string.punctuation]
    return no_punct_seq_tokens


def preprocess_text(text):
    # contaction expasion
    expanded_text = contractions.fix(text)

    # tokenization
    tokens = tokenize_text(expanded_text)

    # lemmatization with pos taging and to lowercase
    tokens = lemmatize_with_pos_text(tokens)

    # negatif tokens
    tokens = negatif_tokens(tokens)

    # stopwords removal
    tokens = remove_stopwords(tokens)

    # most frequent words removal
    tokens = remove_frequent(tokens, 1)

    return ' '.join(tokens)


topics_dict={}
topics_dict[0]="Slow table service"
topics_dict[1]="Bland food"
topics_dict[2]="Terrible food and service"
topics_dict[3]="rude employees"
topics_dict[4]="non delicious chicken and salad"
topics_dict[5]="long wait time"
topics_dict[6]="Expensive and small portions"
topics_dict[7]="Restaurant.com certificates refused"
topics_dict[8]="0 stars"
topics_dict[9]="bad bar service"
topics_dict[10]="Pizza delivery"
topics_dict[11]="bad asian food"
topics_dict[12]="shrimp/sushi"
topics_dict[13]="dirty restaurant"
topics_dict[14]="not organizes"
def predict_topics(model, vectorizer, n_topics, text):
  polarity=TextBlob(text).sentiment.polarity
  if polarity<0:
    text=preprocess_text(text)
    text=[text]
    vectorized=vectorizer.transform(text)
    topics_correlations=model.transform(vectorized)
    unsorted_topics_correlations=topics_correlations[0].copy()
    topics_correlations[0].sort()
    sorted=topics_correlations[0][::-1]
    print(sorted)
    topics=[]
    for i in range(n_topics):
      corr_value= sorted[i]
      result = np.where(unsorted_topics_correlations == corr_value)[0]
      topics.append(topics_dict.get(result[0]))
    returned=[polarity, topics]
    return returned
  else:
    returned=[polarity, False]
    return returned