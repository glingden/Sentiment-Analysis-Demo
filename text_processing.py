

##import libraries
import re
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords, words,  wordnet
from nltk.stem import WordNetLemmatizer



#list of contraction words
contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "can not", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}

# regex patterns for contraction words
contractions_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))  # use key



def expand_contractions(text, contractions):
    """
    Function to expand contraction words into two-words(e.g. 'can't' -->> 'can not')

    Parameters:
       text(str): text to expand contraction words
       contractions(dict): contractions  words in dict type

    Returns:
       Text with out contraction words

    """


    def replace(match_object):
        """
        Parameters:
            match_object: matched words in  regex pattern 'contractions_re'

        Return:
            values for matched words
        """
        return contractions[match_object.group(0)] # get dict value

    return contractions_re.sub(replace, text) # sub with 'replace'


# find wordnet POS-tagging
def get_wordnet_pos(pos_tag):

    """
    Parameters:
        pos_tag: Word POS tag

    Returns:
        wordnet pos tag(for example, 'n','r','j')
    """

    if pos_tag.startswith('J'):
        return wordnet.ADJ

    elif pos_tag.startswith('R'):
        return wordnet.ADV

    else:
        return wordnet.NOUN


# text preprocessing
def text_cleaning(text):
    """
    Function to clean documents

    Parameter:
       text(str): text for preprocessing

    Returns:
        The Clean Processed text

    """

    # 1. convert words to lower case
    text = text.lower()

    # 2.replace contraction words using function - replace_contractions()
    text = expand_contractions(text, contraction_dict)

    # 3.remove alphanumeric characters
    text = re.sub(r"[^A-Za-z]", " ", text)
    text = text.split()

    # 4.remove stop words and words with less than 3 letters
    stops_word = set(stopwords.words("english"))
    text = [w for w in text if w in ['no', 'not'] or (
        w not in stops_word and len(w) >= 3)]  # include ['no', 'not', 'too', 'so']

    # 4.take only noun, adjective and adverb using POS-tag
    noun_adj_adv = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS', 'RB', 'RBR',
                    'RBS']  # inlcude only noun, adjective and adverb
    # text = [word for word in text  if pos_tag([word])[0][1] in noun_adj_adv]


    # 5.lemmatize words based on specific POS-tags
    lema = WordNetLemmatizer()
    lema_words = [lema.lemmatize(word, pos=get_wordnet_pos(pos_tag([word])[0][1])) for word in text if
                  pos_tag([word])[0][1] in noun_adj_adv]
    text = " ".join(lema_words)  # return string

    return text