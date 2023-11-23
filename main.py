import pandas as pd 
import gc
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import nltk
#nltk.download('wordnet')   #run only once 
from nltk.stem import WordNetLemmatizer
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis


file_path = 'full_year.csv'
data = pd.read_csv(file_path)


# we'll use the clean_text
spirituality_df = data[data['spirituality'] == True][['clean_text']]
religion_df = data[data['religion'] == True][['clean_text']]

# remove the wole data variable
del data
gc.collect()

def preprocess(text):
    """
    simple preprocessing function
    """
    result = []
    for token in simple_preprocess(text):
        if token not in STOPWORDS and len(token) > 3:
            result.append(WordNetLemmatizer().lemmatize(token, pos='v'))
    return result

# Apply preprocessing to the clean_text column in each DataFrame
spirituality_texts = spirituality_df['clean_text'].map(preprocess)
religion_texts = religion_df['clean_text'].map(preprocess)

# Create a dictionary and corpus for each data. Check gensim documentation for explanation on what that is
spirituality_dictionary = corpora.Dictionary(spirituality_texts)
spirituality_corpus = [spirituality_dictionary.doc2bow(text) for text in spirituality_texts]

religion_dictionary = corpora.Dictionary(religion_texts)
religion_corpus = [religion_dictionary.doc2bow(text) for text in religion_texts]

# Apply LDA model
num_topics = 10 # we can play around with this
spirituality_lda = models.LdaModel(corpus=spirituality_corpus, num_topics=num_topics, id2word=spirituality_dictionary, passes=10)
religion_lda = models.LdaModel(corpus=religion_corpus, num_topics=num_topics, id2word=religion_dictionary, passes=10)

# print the topics in the LDA model
for idx, topic in spirituality_lda.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

for idx, topic in religion_lda.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))



# Visualize and save the spirituality topics visualization
spirituality_vis = gensimvis.prepare(spirituality_lda, spirituality_corpus, spirituality_dictionary)
pyLDAvis.display(spirituality_vis)
pyLDAvis.save_html(spirituality_vis, 'spirituality_lda_visualization.html')

# same for the religion with visualisation
religion_vis = gensimvis.prepare(religion_lda, religion_corpus, religion_dictionary)
pyLDAvis.display(religion_vis)
pyLDAvis.save_html(religion_vis, 'religion_lda_visualization.html')
