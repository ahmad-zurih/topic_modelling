import pandas as pd
import gc
from bertopic import BERTopic
from nltk.stem import WordNetLemmatizer
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import os


os.environ["TOKENIZERS_PARALLELISM"] = "true"


# File path
file_path = '/home/ahmad-unibe/topic_modelling_presentation/full_year.csv'
data = pd.read_csv(file_path)

# Filter data based on 'spirituality' and 'religion'
spirituality_df = data[data['spirituality'] == True][['clean_text']]
religion_df = data[data['religion'] == True][['clean_text']]

# Clean up to free memory
del data
gc.collect()

def preprocess(text):
    """
    Simple preprocessing function
    """
    result = []
    for token in simple_preprocess(text):
        if token not in STOPWORDS and len(token) > 3:
            result.append(WordNetLemmatizer().lemmatize(token, pos='v'))
    return ' '.join(result)  # Join the tokens back into a string

# Apply preprocessing to the clean_text column
spirituality_texts = spirituality_df['clean_text'].map(preprocess).tolist()
religion_texts = religion_df['clean_text'].map(preprocess).tolist()

# Initialize BERTopic models with specified device
spirituality_topic_model = BERTopic(verbose=True)
religion_topic_model = BERTopic(verbose=True)


# Fit the models
spirituality_topics, _ = spirituality_topic_model.fit_transform(spirituality_texts)
religion_topics, _ = religion_topic_model.fit_transform(religion_texts)

# Reduce the number of topics in BERTopic to align with LDA
spirituality_topic_model = spirituality_topic_model.reduce_topics(spirituality_texts, nr_topics=30)

# Reduce the number of topics in BERTopic to align with LDA
religion_topic_model = religion_topic_model.reduce_topics(religion_texts, nr_topics=30)

# Print topics
print(spirituality_topic_model.get_topic_info())
print(religion_topic_model.get_topic_info())



# Visualize topics and save the visualization
spirituality_vis = spirituality_topic_model.visualize_topics()
spirituality_vis.write_html('spirituality_topics_visualization.html')

religion_vis = religion_topic_model.visualize_topics()
religion_vis.write_html('religion_topics_visualization.html')


# Assuming the reduction has been done and you have `spirituality_topic_model` ready

# Get top 10 topics based on the number of documents
top_n_topics = spirituality_topic_model.get_topic_info().head(20)  # includes -1 outlier topic

# Filter the topics to include only the top 10 (excluding -1 if present)
top_n_topic_nums = top_n_topics[top_n_topics.Topic != -1]['Topic'].values[:20]

# Create a visualization only for the top 10 topics
spirituality_vis = spirituality_topic_model.visualize_topics(topics=top_n_topic_nums)
spirituality_vis.write_html('top_10_spirituality_topics_visualization.html')

# Repeat the process for the religion model
# Get top 10 topics based on the number of documents
top_n_topics = religion_topic_model.get_topic_info().head(20)  # includes -1 outlier topic

# Filter the topics to include only the top 10 (excluding -1 if present)
top_n_topic_nums = top_n_topics[top_n_topics.Topic != -1]['Topic'].values[:20]

# Create a visualization only for the top 10 topics
religion_vis = religion_topic_model.visualize_topics(topics=top_n_topic_nums)
religion_vis.write_html('top_10_religion_topics_visualization.html')
