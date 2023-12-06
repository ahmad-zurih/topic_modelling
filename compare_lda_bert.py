import pandas as pd
import gc
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer
from bertopic import BERTopic
import matplotlib.pyplot as plt
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

# BERTopic Model
bertopic_model = BERTopic(verbose=True)
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

# Compare Topics and Print Common Words
def compare_topics(lda_topics, bertopic_topics):
    similarity_matrix = []
    for lda_idx, lda_topic in lda_topics.items():
        for bertopic_idx, bertopic_topic in bertopic_topics.items():
            similarity = jaccard_similarity(lda_topic, bertopic_topic)
            common_words = set(lda_topic).intersection(bertopic_topic)
            similarity_matrix.append((lda_idx, bertopic_idx, similarity, common_words))
    return sorted(similarity_matrix, key=lambda x: x[2], reverse=True)

# Run comparison and report the topics with common words
topic_similarities = compare_topics(lda_topics, bertopic_topics)
for lda_idx, bertopic_idx, sim, common_words in topic_similarities:
    print(f"LDA Topic {lda_idx} and BERTopic Topic {bertopic_idx} have a similarity score of: {sim}")
    print(f"Common Words: {', '.join(common_words)}\n")



# Filter out topic pairs with zero similarity
filtered_topic_similarities = [item for item in topic_similarities if item[2] > 0]

# Extract data for the bar chart
lda_indices, bertopic_indices, similarities, common_words_list = zip(*filtered_topic_similarities)
unique_lda_indices = sorted(set(lda_indices))
bar_width = 0.35  # the width of the bars

# Create a bar chart
fig, ax = plt.subplots(figsize=(15, 7))

# Generate bar positions
lda_bar_positions = list(range(len(unique_lda_indices)))
bertopic_bar_positions = [x + bar_width for x in lda_bar_positions]

# Gather the highest similarity score for each unique LDA topic
lda_bars = []
bertopic_bars = []
for lda_idx in unique_lda_indices:
    lda_similarities = [item[2] for item in filtered_topic_similarities if item[0] == lda_idx]
    bertopic_similarities = [item[2] for item in filtered_topic_similarities if item[1] == lda_idx]
    # Take the max similarity for each LDA topic, if there are multiple
    lda_bars.append(max(lda_similarities) if lda_similarities else 0)
    bertopic_bars.append(max(bertopic_similarities) if bertopic_similarities else 0)

# Plot each set of bars
lda_bars_plot = ax.bar(lda_bar_positions, lda_bars, bar_width, label='LDA')
bertopic_bars_plot = ax.bar(bertopic_bar_positions, bertopic_bars, bar_width, label='BERTopic')

# Add some text for labels, title, and axes ticks
ax.set_xlabel('Topic Index')
ax.set_ylabel('Similarity Score')
ax.set_title('Topic Similarity Scores between LDA and BERTopic')
ax.set_xticks([index + bar_width / 2 for index in range(len(unique_lda_indices))])
ax.set_xticklabels(unique_lda_indices)
ax.legend()

# Function to add a label above each bar
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        if height > 0:  # Only label bars with a height > 0 (i.e., similarity > 0)
            ax.annotate('{}'.format(round(height, 2)),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

# Call the function to add labels
add_labels(lda_bars_plot)
add_labels(bertopic_bars_plot)

# Show the plot
plt.tight_layout()
plt.show()


# Cleanup
del data
gc.collect()
