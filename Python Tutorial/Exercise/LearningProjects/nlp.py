import pandas as pd
import re 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer



# read json into a dataframe
df_idf=pd.read_json("stackoverflow-data-idf.json",lines=True)
 
# # print schema
# print("Schema:\n\n",df_idf.dtypes)
# print("Number of questions,columns=",df_idf.shape)

# # print("Body:\n\n",df_idf.body)
# print("Body:\n\n",df_idf.title)    

###
def pre_process(text):
    
    # lowercase
    text=text.lower()
    
    #remove tags
    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
    
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    return text
 
df_idf['text'] = df_idf['title'] + df_idf['body']
df_idf['text'] = df_idf['text'].apply(lambda x:pre_process(x))
# get test docs into a list
df_test=pd.read_json("stackoverflow-test.json",lines=True)
df_test['text'] = df_test['title'] + df_test['body']
df_test['text'] =df_test['text'].apply(lambda x:pre_process(x))

docs_test=df_test['text'].tolist()
docs_title=df_test['title'].tolist()
docs_body=df_test['body'].tolist()

 
#show the second 'text' just for fun

# print df_idf['text'][2]

# For Practice
# file = open('stopWords.txt', "r")
# for line in file:
#    print line,

def get_stop_words(stop_file_path):
    """load stop words """
    
    with open(stop_file_path, 'r') as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return frozenset(stop_set)
 
#load a set of stop words
stopwords=get_stop_words("stopWords.txt")
# print("Stopwrords ", stopwords)
 
#get the text column 
docs=df_idf['text'].tolist()
 
#create a vocabulary of words, 
#ignore words that appear in 85% of documents, 
#eliminate stop words
cv=CountVectorizer(max_df=0.85,stop_words=stopwords,max_features=10000)
word_count_vector=cv.fit_transform(docs)
# print 'Word Count Vector ','\n',word_count_vector
# print "Cv ", cv
# list(word_count_vector)[: 10]

cvList= list(cv.vocabulary_.keys())
# print "CV list is \n",cvList[:10]

# print  list(cv.get_feature_names())[2000:2015]

#########################
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        fname = feature_names[idx]
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results

########################

# you only needs to do this once
feature_names=cv.get_feature_names()

# get the document that we want to extract keywords from
doc=docs_test[0]

#generate tf-idf for the given document
tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))

#sort the tf-idf vectors by descending order of scores
sorted_items=sort_coo(tf_idf_vector.tocoo())

#extract only the top n; n here is 10
keywords=extract_topn_from_vector(feature_names,sorted_items,10)

# now print the results
print("\n=====Title=====")
print(docs_title[0])
print("\n=====Body=====")
print(docs_body[0])
print("\n===Keywords===")
for k in keywords:
    print(k,keywords[k])

###########################################

# put the common code into several methods
def get_keywords(idx):

    #generate tf-idf for the given document
    tf_idf_vector=tfidf_transformer.transform(cv.transform([docs_test[idx]]))

    #sort the tf-idf vectors by descending order of scores
    sorted_items=sort_coo(tf_idf_vector.tocoo())

    #extract only the top n; n here is 10
    keywords=extract_topn_from_vector(feature_names,sorted_items,10)
    
    return keywords

def print_results(idx,keywords):
    # now print the results
    print("\n=====Title=====")
    print(docs_title[idx])
    print("\n=====Body=====")
    print(docs_body[idx])
    print("\n===Keywords===")
    for k in keywords:
        print(k,keywords[k])

# ########################################    