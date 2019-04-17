## Checkout this article http://kavita-ganesan.com/text-preprocessing-tutorial/#.XLbFCzAzaM9
## GitHub: https://github.com/kavgan/nlp-in-practice/blob/master/text-pre-processing/Text%20Preprocessing%20Examples.ipynb
import nltk
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re

# nltk.download('wordnet')
#  nltk.download('wordnet')

# Lowercasing
texts=["CANADA","Canada","canadA","canada"]
lower_words=[word.lower() for word in texts]

#print lower_words

# Stemming

# init stemmer
porter_stemmer=PorterStemmer()

# stem connect variations
#print "\n\nstem connect variations"
words=["connect","connected","connection","connections","connects"]
stemmed_words=[porter_stemmer.stem(word=word) for word in words]

stemdf= pd.DataFrame({'original_word': words,'stemmed_word': stemmed_words})
#print stemdf

# stem trouble variations
#print "\n\nstem trouble variations"

words=["trouble","troubled","troubles","troublemsome"]
stemmed_words=[porter_stemmer.stem(word=word) for word in words]

stemdf= pd.DataFrame({'original_word': words,'stemmed_word': stemmed_words})
#print stemdf

## Lemmatizer
lemmatizer = WordNetLemmatizer()
#lemmatize trouble variations
#print "\n\nlemmatize trouble variations"
words=["trouble","troubling","troubled","troubles",]
lemmatized_words=[lemmatizer.lemmatize(word=word,pos='v') for word in words]
lemmatizeddf= pd.DataFrame({'original_word': words,'lemmatized_word': lemmatized_words})
lemmatizeddf=lemmatizeddf[['original_word','lemmatized_word']]
#print lemmatizeddf


## StopWords Removal
stopwords=['this','that','and','a','we','it','to','is','of','up','need']
text="this is a text full of content and we need to clean it up"
words=text.split(" ")
shortlisted_words=[]

#remove stop words
#print "\n\nremove stop words"
for w in words:
    if w not in stopwords:
        shortlisted_words.append(w)
    else:
        shortlisted_words.append("W")

#print("original sentence = ",text)    
#print("sentence with stop words removed= ",' '.join(shortlisted_words))

## Noise Removal
# stem raw words with noise
raw_words=["..trouble..","trouble<","trouble!","<a>trouble</a>",'1.trouble']
stemmed_words=[porter_stemmer.stem(word=word) for word in raw_words]
stemdf= pd.DataFrame({'raw_word': raw_words,'stemmed_word': stemmed_words})
# print stemdf

def scrub_words(text):
    """Basic cleaning of texts."""
    
    # remove html markup
    text=re.sub("(<.*?>)","",text)
    
    #remove non-ascii and digits
    text=re.sub("(\\W|\\d)"," ",text)
    
    #remove whitespace
    text=text.strip()
    return text

# stem words already cleaned
cleaned_words=[scrub_words(w) for w in raw_words]
cleaned_stemmed_words=[porter_stemmer.stem(word=word) for word in cleaned_words]
stemdf= pd.DataFrame({'raw_word': raw_words,'cleaned_word':cleaned_words,'stemmed_word': cleaned_stemmed_words})
stemdf=stemdf[['raw_word','cleaned_word','stemmed_word']]
print stemdf