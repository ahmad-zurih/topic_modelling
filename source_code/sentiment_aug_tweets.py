import argparse
import csv
import os


data_dir = '/home/janis/ownCloud/P8/P8 Data Exchange/'
fname_sentiment = 'sentiment_raw_results.txt'
fname_tweets = '8set_ALL.name_text_source_ASCII_cleaned.txt'
fname_out = '8set_ALL.name_text_source_ASCII_cleaned_w_sentiment.txt'

fsentiment = open(os.path.join(data_dir, fname_sentiment))
ftweets = open(os.path.join(data_dir, fname_tweets))
fout = open(os.path.join(data_dir, fname_out), 'w')

sentiment_reader = csv.reader(fsentiment, delimiter='\t')
tweet_reader = csv.reader(ftweets, delimiter='\t', quoting=csv.QUOTE_ALL)
writer = csv.writer(fout, delimiter='\t', quoting=csv.QUOTE_ALL)

for sentiment, tweet in zip(sentiment_reader, tweet_reader):
    if sentiment[0] == 'POSITIVE':
        tweet[1] += ' sentipositive'
    elif sentiment[0] == 'NEGATIVE':
        tweet[1] += ' sentinegative'
    else:
        raise Exception(f'Row contains unexpected sentiment: {sentiment}')
    writer.writerow(tweet)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('')