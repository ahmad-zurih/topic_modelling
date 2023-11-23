import pandas as pd
import gc
from bertopic import BERTopic
from nltk.stem import WordNetLemmatizer
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import os


os.environ["TOKENIZERS_PARALLELISM"] = "false"


# File path
file_path = 'full_year.csv'
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
spirituality_topic_model = BERTopic()
religion_topic_model = BERTopic()

# Fit the models
spirituality_topics, _ = spirituality_topic_model.fit_transform(spirituality_texts)
religion_topics, _ = religion_topic_model.fit_transform(religion_texts)

# Print topics
print(spirituality_topic_model.get_topic_info())
print(religion_topic_model.get_topic_info())



# Visualize topics and save the visualization
spirituality_vis = spirituality_topic_model.visualize_topics()
spirituality_vis.write_html('spirituality_topics_visualization.html')

religion_vis = religion_topic_model.visualize_topics()
religion_vis.write_html('religion_topics_visualization.html')

