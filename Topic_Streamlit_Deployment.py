import numpy as np
import pickle
import pandas as pd
import streamlit as st
import tweepy
import pandas as pd
import re
import emoji
import nltk
import datetime
import spacy
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import time
st.set_page_config(layout="wide")
pickle_in=open("topic_modelling.pkl","rb")
topic_modelling=pickle.load(pickle_in)

def Predict_Topics(cv_arr,vocab_tf_idf):
    result=pd.DataFrame()
    from sklearn.decomposition import LatentDirichletAllocation
    lda_model = LatentDirichletAllocation(n_components = 10, max_iter = 20)
    X_topics = lda_model.fit_transform(cv_arr)
    topic_words = lda_model.components_
    n_top_words = 10
    for i, topic_dist in enumerate(topic_words):
        sorted_topic_dist = np.argsort(topic_dist)
        topic_words = np.array(vocab_tf_idf)[sorted_topic_dist]
        topic_words = topic_words[:-n_top_words:-1]
        result=result.append({'topics':str(i+1),'topics_words':topic_words},ignore_index=True)
    result.reset_index(drop = True, inplace=True)
    return result

def collecting_data (Date1,Date2):
    mykeys = open('tweetfile.txt','r').read().splitlines()
    api_key = mykeys[0]
    api_key_secret = mykeys[1]
    access_token = mykeys[2]
    access_token_secret = mykeys[3]
    auth = tweepy.OAuthHandler(consumer_key = api_key, consumer_secret = api_key_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth,wait_on_rate_limit=True)
    search_words="news"
    date_since=Date1
    data_until=Date2
    tweets = tweepy.Cursor(api.search,q=search_words,lang="en",tweet_mode='extended',since=date_since,until=data_until,result_type="recent").items(300)
    tweets = tweepy.Cursor(api.search,
              q=search_words,
              lang="en",
              since=date_since).items(300)
    time.sleep(5) 
    st.success('Data is collected Successfully ')   
    return tweets

def clean_data(tweets):
    st.success('Please Wait, data cleaning is in process')
    s=[] 
    for tweet in tweets:
        s.append(tweet.text)
    df=pd.DataFrame({'tweet':s})
    import nltk       
    words = set(nltk.corpus.words.words())
    tweet=np.array(df.tweet)
    cleaned_tweet=[]
    for i in df.tweet:
        no_punc_text = i.translate(str.maketrans('', '', string.punctuation))
        no_punc_text=re.sub("(RT)?(ht)?", "", no_punc_text) # to remove RT and ht word
        no_punc_text1=re.sub("[\W\d]", " ", no_punc_text) #to remove not word character and numbers
        no_punc_text2=re.sub("[^a-zA-Z]", " ", no_punc_text1) #to remove forien language word character
        no_punc_text2=" ".join(w for w in nltk.wordpunct_tokenize(no_punc_text2) if w.lower() in words or not w.isalpha())
        cleaned_tweet.append(no_punc_text2)
    df['cleaned_tweet']=cleaned_tweet
    df1=df.copy() 
    corpus=df1.cleaned_tweet.unique()
    return corpus

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    from nltk.corpus import wordnet
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,"N": wordnet.NOUN,"V": wordnet.VERB,"R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def clean(doc):
    stop = set(stopwords.words('english'))
    stop.update(["new","news",'via','take','first','one','say','time','big','see','come','good','another','today','make','get','great','could','like','make','set','end','dont'])
    exclude = set(string.punctuation) 
    lemma = WordNetLemmatizer() 
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)  
    normalized = " ".join(lemma.lemmatize(word,get_wordnet_pos(word)) for word in punc_free.split())  
    return normalized

def remove_two_char_words (clean_corpus):
    corpus1=[]
    for i in clean_corpus:
        doc=[]
        #j=i.split()
        for z in i:
            #print(len(z))
            if len(z)>2:
                doc.append(z)
        #print(doc)
        doc=" ".join(doc)
        doc1=doc.split()
        #print(doc1)
        corpus1.append(doc1)
    st.success('Data is cleaned successfully')
    time.sleep(5)
    return corpus1

def model_inputs (clean_corpus):
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    tf_idf_vectorizer = TfidfVectorizer(tokenizer=lambda doc: doc, lowercase=False)
    cv_vectorizer = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)
    tf_idf_arr = tf_idf_vectorizer.fit_transform(clean_corpus)
    cv_arr = cv_vectorizer.fit_transform(clean_corpus)
    vocab_tf_idf = tf_idf_vectorizer.get_feature_names()
    vocab_cv = cv_vectorizer.get_feature_names()
    return cv_arr, vocab_tf_idf

def new_clean_corpus(clean_corpus):
    abc = []  #to create single list
    for i in clean_corpus:
        abc.append(' '.join(i))                                    
    abc2=" ".join(abc)    
    nlp = spacy.load('en_core_web_sm')                
    one_block = abc2
    doc_block = nlp(one_block)
    final_corpus = [token.text for token in doc_block if token.pos_ in ('PROPN','X','NOUN','ADJ')]
    imp_words = set(final_corpus)
    corpus1=[]
    for i in clean_corpus:
        doc=[]
        for z in i:
            if z in imp_words:
                doc.append(z)
        doc=" ".join(doc)
        doc1=doc.split()
        corpus1.append(doc1)
    return corpus1  

def main():
    import numpy as np
    import pickle
    import pandas as pd
    import streamlit as st
    import tweepy
    import pandas as pd
    import re
    import emoji
    import nltk
    import datetime
    import spacy
    from nltk.corpus import stopwords 
    from nltk.stem.wordnet import WordNetLemmatizer
    import string
    import time
    st.subheader("(Topic_Modelling App)")
    html_temp="""    <div style="background-color:blue;padding:10px">
    <h1 style="color:white;text-align:center;">Trending Topics</h1>
    </div>"""
    st.markdown(html_temp,unsafe_allow_html=True)    
    Date1 = st.sidebar.date_input('start date', datetime.date.today()-datetime.timedelta(days=7))
    Date2 = st.sidebar.date_input('end date', datetime. date.today())
    
    if st.sidebar.button('Find Topics'):
        with st.empty():
            st.success('Please Wait, till live data is extracted from Tweeter')
            time.sleep(20)
            tweets = collecting_data (Date1,Date2)
            corpus = clean_data(tweets)
            clean_corpus = [clean(doc).split() for doc in corpus]
            clean_corpus = remove_two_char_words (clean_corpus)
            clean_corpus = new_clean_corpus(clean_corpus)
            cv_arr, vocab_tf_idf = model_inputs (clean_corpus)
            st.success('Finding Topics is in process')
            time.sleep(10)
            result=" "
            result = Predict_Topics(cv_arr, vocab_tf_idf) 
            st.write(result)
               
if __name__ == "__main__":
    main()