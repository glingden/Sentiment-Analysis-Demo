import re
import string
from nltk import pos_tag
from nltk.corpus import stopwords,  wordnet
from nltk.stem import WordNetLemmatizer

from contraction_words import contraction_dict


# Find wordnet POS-tagging
def get_wordnet_pos(pos_tag:str)-> str:

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



def text_cleaning(text:str)-> str:
    """
    Function to clean documents

    Parameter:
       text(str): text for preprocessing

    Returns:
        The Clean Processed text
    """
    # Replace contractions with their expanded forms
    for word in text.split():
        if word.lower() in contraction_dict:
            text = text.replace(word, contraction_dict[word.lower()])
            
   
    # Convert to lowercase
    text = text.lower()

    # Remove non-alphanumeric characters
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
  

    # Tokenize and remove stop words
    words = [w for w in text.split() if w not in stopwords.words('english')]
   

    # Stem the words
    # stemmed_words = [stemmer.stem(word) for word in words]

    # Take only noun, adjective and adverb using POS-tag
    pos_tags = pos_tag(words)
    noun_adj_adv = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']
    filtered_words = [word for word, pos in pos_tags if pos in noun_adj_adv]


    # Lemmatize words based on specific POS-tags
    lema = WordNetLemmatizer()
    lema_words = [lema.lemmatize(word, pos=get_wordnet_pos(pos_tag([word])[0][1])) for word in filtered_words if
                  pos_tag([word])[0][1] in noun_adj_adv]
    
    # Join the words back into a string
    text = ' '.join(lema_words)

    return text


