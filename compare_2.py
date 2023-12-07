import pandas as pd
import gc
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer
from bertopic import BERTopic
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

# Enable parallelism in tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Define the preprocessing function
def preprocess(text):
    result = []
    for token in simple_preprocess(text):
        if token not in STOPWORDS and len(token) > 3 and token != "embeddedurl":
            result.append(WordNetLemmatizer().lemmatize(token, pos='v'))
    return ' '.join(result)

# Load Data
file_path = 'full_year.csv'
data = pd.read_csv(file_path)

# Preprocess Data
data['processed_text'] = data['clean_text'].apply(preprocess)

# Split the data into two datasets for spirituality and religion
texts = data['processed_text'].tolist()

# Gensim LDA Model
dictionary = corpora.Dictionary(data['processed_text'].apply(lambda x: x.split()))
corpus = [dictionary.doc2bow(text.split()) for text in data['processed_text']]
lda_model = models.LdaModel(corpus=corpus, num_topics=10, id2word=dictionary, passes=10)
lda_topics = {i: [word for word, _ in lda_model.show_topic(i, topn=10)] for i in range(lda_model.num_topics)}

# BERTopic Model with K-Means
cluster_model = KMeans(n_clusters=10, random_state=0)
bertopic_model = BERTopic(verbose=True, hdbscan_model=cluster_model)
topics, _ = bertopic_model.fit_transform(texts)

# Reduce the number of topics in BERTopic to align with LDA
bertopic_model = bertopic_model.reduce_topics(texts, nr_topics=10)

# Extract top 10 words for each topic from BERTopic
bertopic_topics = {topic_idx: [word[0] for word in bertopic_model.get_topic(topic_idx)[:10]] for topic_idx in bertopic_model.get_topics()}




# Compare Topics
def jaccard_similarity(set1, set2):
    intersection = len(set(set1).intersection(set2))
    union = len(set(set1).union(set2))
    return intersection / union if union != 0 else 0

# Compare Topics and Print Common and Uncommon Words
def compare_topics(lda_topics, bertopic_topics):
    similarity_matrix = []
    for lda_idx, lda_topic in lda_topics.items():
        for bertopic_idx, bertopic_topic in bertopic_topics.items():
            similarity = jaccard_similarity(lda_topic, bertopic_topic)
            common_words = set(lda_topic).intersection(bertopic_topic)
            uncommon_words_lda = set(lda_topic).difference(bertopic_topic)
            uncommon_words_bertopic = set(bertopic_topic).difference(lda_topic)
            similarity_matrix.append((lda_idx, bertopic_idx, similarity, common_words, uncommon_words_lda, uncommon_words_bertopic))
    return sorted(similarity_matrix, key=lambda x: x[2], reverse=True)

# Run comparison and report the topics with common and uncommon words
topic_similarities = compare_topics(lda_topics, bertopic_topics)
for lda_idx, bertopic_idx, sim, common_words, uncommon_words_lda, uncommon_words_bertopic in topic_similarities:
    print(f"LDA Topic {lda_idx} and BERTopic Topic {bertopic_idx} have a similarity score of: {sim}")
    print(f"Common Words: {', '.join(common_words)}")
    print(f"Uncommon Words in LDA: {', '.join(uncommon_words_lda)}")
    print(f"Uncommon Words in BERTopic: {', '.join(uncommon_words_bertopic)}\n")


# Cleanup
del data
gc.collect()
