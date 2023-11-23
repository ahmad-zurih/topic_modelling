import pandas as pd 
import gc
from gensim import corpora, models
from gensim.models.phrases import Phrases, Phraser
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import nltk
from nltk.stem import WordNetLemmatizer
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

#nltk.download('wordnet')   #run only once 

file_path = 'full_year.csv'
data = pd.read_csv(file_path)

# We'll use the clean_text
spirituality_df = data[data['spirituality'] == True][['clean_text']]
religion_df = data[data['religion'] == True][['clean_text']]

# Remove the whole data variable to free up memory
del data
gc.collect()

# Preprocess function to tokenize the text
def preprocess(texts):
    for text in texts:
        yield simple_preprocess(str(text), deacc=True)  # deacc=True removes punctuations

# Apply preprocessing to the clean_text column in each DataFrame
spirituality_texts = list(preprocess(spirituality_df['clean_text']))
religion_texts = list(preprocess(religion_df['clean_text']))

# Create bigrams
bigram_phrases_spirituality = Phrases(spirituality_texts, min_count=3, threshold=50)
bigram_phrases_religion = Phrases(religion_texts, min_count=3, threshold=50)

bigram_spirituality = Phraser(bigram_phrases_spirituality)
bigram_religion = Phraser(bigram_phrases_religion)

# Function to form bigrams
def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]

# Apply bigrams to the texts
spirituality_texts = make_bigrams(spirituality_texts, bigram_spirituality)
religion_texts = make_bigrams(religion_texts, bigram_religion)

# Function for lemmatization and removing stopwords
def lemmatize_and_remove_stopwords(texts):
    result = []
    for text in texts:
        lemmatized_text = [WordNetLemmatizer().lemmatize(token, pos='v') for token in text if token not in STOPWORDS and len(token) > 3]
        result.append(lemmatized_text)
    return result

# Apply lemmatization and remove stopwords
spirituality_texts = lemmatize_and_remove_stopwords(spirituality_texts)
religion_texts = lemmatize_and_remove_stopwords(religion_texts)

# Create a dictionary and corpus for each DataFrame
spirituality_dictionary = corpora.Dictionary(spirituality_texts)
spirituality_corpus = [spirituality_dictionary.doc2bow(text) for text in spirituality_texts]

religion_dictionary = corpora.Dictionary(religion_texts)
religion_corpus = [religion_dictionary.doc2bow(text) for text in religion_texts]

# Apply LDA model
num_topics = 10  # You can adjust this
spirituality_lda = models.LdaModel(corpus=spirituality_corpus, num_topics=num_topics, id2word=spirituality_dictionary, passes=10)
religion_lda = models.LdaModel(corpus=religion_corpus, num_topics=num_topics, id2word=religion_dictionary, passes=10)

# Print the topics in the LDA model
for idx, topic in spirituality_lda.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

for idx, topic in religion_lda.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

# Visualize and save the spirituality topics visualization
spirituality_vis = gensimvis.prepare(spirituality_lda, spirituality_corpus, spirituality_dictionary)
pyLDAvis.display(spirituality_vis)
pyLDAvis.save_html(spirituality_vis, 'spirituality_lda_visualization_bigrams.html')

# Same for the religion topics visualization
religion_vis = gensimvis.prepare(religion_lda, religion_corpus, religion_dictionary)
pyLDAvis.display(religion_vis)
pyLDAvis.save_html(religion_vis, 'religion_lda_visualization_bigrams.html')
