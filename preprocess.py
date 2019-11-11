# helper functions

#imports
import re  # regular expressions
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer  # lemmatizer
from nltk.stem import PorterStemmer  # stemmer
from nltk.tokenize import word_tokenize  # tokenizer

# preprocess text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# check blank string function
def isBlank(myString):
    myString = str(myString)
    return not (myString and myString.strip())

# tokenizers (a lot of feature vectors, raw)
def tokenizeBasic(txt):
    return txt.split()

# lemmatizer (doesn't remove punctuation)
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


# stemmer (no difference from lemmatizer?)
stemmer = PorterStemmer()
analyzer = CountVectorizer().build_analyzer()

def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))

#TODO

def cleanUp(text):
    # Initilaise Lemmatizer
    lemm = WordNetLemmatizer()

    # use alternative stemmer
    #snowball = SnowballStemmer(language = 'english')
    #ps = PorterStemmer()

    # load stopwords
    #my_stopwords = stopwords.words('english')
    my_stopwords = []
    clean_text = ""
    # tokenize words (convert text from byte to string)
    words = word_tokenize(str(text, errors="ignore"))
    # print(words[:8])

    for word in words:

        w = lemm.lemmatize(word.lower())
        #w = re.sub('<.*?>', '', w) # remove HTML tags
        #w = re.sub(r'[^\w\s</>]', '', w) # remove punc.
        w = re.sub(r'\d+','',w)# remove numbers
        # lemmatize the word(normalized to lower case)
        
        # stem the word
        #w = snowball.stem(w.lower())

        # print(w)

        # filter out stopwords
        if w not in my_stopwords and len(w) > 0:
          clean_text += w + " "

    return clean_text

def cleanUpPP(text):
    # Initilaise Lemmatizer
    lemm = WordNetLemmatizer()

    # use alternative stemmer
    #snowball = SnowballStemmer(language = 'english')
    #ps = PorterStemmer()

    # load stopwords
    my_stopwords = stopwords.words('english')
    my_stopwords = []
    clean_text = ""
    # tokenize words (convert text from byte to string)
    words = word_tokenize(str(text, errors="ignore"))
    # print(words[:8])

    for word in words:

        w = lemm.lemmatize(word.lower())
        #w = re.sub('<.*?>', '', w) # remove HTML tags
        w = re.sub(r'[^\w\s</>]', '', w) # remove punc.
        w = re.sub(r'\d+','',w)# remove numbers
        # lemmatize the word(normalized to lower case)
        
        # stem the word
        #w = snowball.stem(w.lower())

        # print(w)

        # filter out stopwords
        if w not in my_stopwords and len(w) > 0:
          clean_text += w + " "

    return clean_text


def preprocess(text):
    clean_data = []
    for x in (text[:][0]): #this is Df_pd for Df_np (text[:])
        new_text = re.sub('<.*?>', '', x)   # remove HTML tags
        new_text = re.sub(r'[^\w\s]', '', new_text) # remove punc.
        new_text = re.sub(r'\d+','',new_text)# remove numbers
        new_text = new_text.lower() # lower case, .upper() for upper          
        if new_text != '':
            clean_data.append(new_text)
    return clean_data

def tokenization_w(words):
    w_new = []
    for w in (words[:][0]):  # for NumPy = words[:]
        w_token = word_tokenize(w)
        if w_token != '':
            w_new.append(w_token)
    return w_new

snowball = SnowballStemmer(language = 'english')
def stemming(words):
    new = []
    stem_words = [snowball.stem(x) for x in (words[:][0])]
    new.append(stem_words)
    return new
    
lemmatizer = WordNetLemmatizer()
def lemmatization(words):
    new = []
    lem_words = [lemmatizer.lemmatize(x) for x in (words[:][0])]
    new.append(lem_words)
    return new