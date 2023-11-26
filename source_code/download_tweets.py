import argparse
import csv
import datetime
import dateutil.parser
import unicodedata
import json
import os
import pandas as pd
import pickle
import requests
import sys
import time
from typing import Dict, List, Tuple


# https://towardsdatascience.com/an-extensive-guide-to-collecting-tweets-from-twitter-api-v2-for-academic-research-using-python-3-518fcb71df2a


def create_headers(bearer_token):
    headers = {'Authorization': 'Bearer {}'.format(bearer_token)}
    return headers


def auth():
    with open('../twitter_tokens.json') as fin:
        return json.load(fin)['Bearer Token']


def create_url(keyword, start_date, end_date, max_results):
    
    search_url = 'https://api.twitter.com/2/tweets/search/all' #Change to the endpoint you want to collect data from

    # change params based on the endpoint you are using
    
    # TODO: <-
    # https://auth0.com/blog/how-to-make-a-twitter-bot-in-python-using-tweepy/
    # https://stackoverflow.com/questions/38717816/twitter-api-text-field-value-is-truncated
    query_params = {'query': keyword,
                    'start_time': start_date,
                    'end_time': end_date,
                    'max_results': max_results,
                    'expansions': 'author_id,in_reply_to_user_id,geo.place_id',
                    'tweet.fields': 'id,text,author_id,geo,created_at,lang,in_reply_to_user_id,conversation_id,public_metrics,referenced_tweets,reply_settings,source',
                    'user.fields': 'id,name,username,created_at,description,public_metrics,verified',
                    'place.fields': 'full_name,id,country,country_code,geo,name,place_type',
                    'next_token': {}}
    return (search_url, query_params)


def connect_to_endpoint(url, headers, params, next_token = None):
    params['next_token'] = next_token   # params object received from create_url function
    response = requests.request('GET', url, headers = headers, params = params)
    print('Endpoint Response Code: ' + str(response.status_code))
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()


def load_dates(fpath: str) -> List[Tuple[str, str]]:
    with open(fpath, 'rb') as fin:
        return pickle.load(fin)


def to_time_windows(times: Dict[int, List[datetime.datetime]]) -> List[Tuple[datetime.datetime, datetime.datetime]]:
    time_windows = []
    for month in times:
        for time in times[month]:
            time_windows.append((time, time + datetime.timedelta(hours=1)))
    return time_windows


def test_request(max_results=500):
    bearer_token = auth()
    headers = create_headers(bearer_token)
    keyword = 'religion OR spirituality lang:en'
    start_time = '2021-03-01T00:00:00.000Z'
    end_time = '2021-03-31T00:00:00.000Z'
    max_results = max_results
    url = create_url(keyword, start_time, end_time, max_results)
    json_response = connect_to_endpoint(url[0], headers, url[1])
    df = pd.DataFrame(json_response['data'])
    df.to_csv('results/downloaded_tweets.csv')


def prepare_output_file(path_out):
    # Create file
    csv_file = open(path_out, 'a', newline='', encoding='utf-8')
    writer = csv.writer(csv_file)

    #Create headers for the data you want to save, in this example, we only want save these columns in our dataset
    writer.writerow(['author id', 'created_at', 'geo', 'id','lang', 'like_count', 'quote_count', 'reply_count','retweet_count','source','tweet'])
    csv_file.close()


def append_to_csv(json_response, fileName):

    #A counter variable
    counter = 0

    #Open OR create the target CSV file
    csvFile = open(fileName, 'a', newline='', encoding='utf-8')
    csvWriter = csv.writer(csvFile)

    #Loop through each tweet
    for tweet in json_response['data']:
        
        # We will create a variable for each since some of the keys might not exist for some tweets
        # So we will account for that

        # 1. Author ID
        author_id = tweet['author_id']

        # 2. Time created
        created_at = dateutil.parser.parse(tweet['created_at'])

        # 3. Geolocation
        if ('geo' in tweet):   
            geo = tweet['geo']['place_id']
        else:
            geo = ' '

        # 4. Tweet ID
        tweet_id = tweet['id']

        # 5. Language
        lang = tweet['lang']

        # 6. Tweet metrics
        retweet_count = tweet['public_metrics']['retweet_count']
        reply_count = tweet['public_metrics']['reply_count']
        like_count = tweet['public_metrics']['like_count']
        quote_count = tweet['public_metrics']['quote_count']

        # 7. source
        source = tweet['source']

        # 8. Tweet text
        text = tweet['text']
        
        # Assemble all data in a list
        res = [author_id, created_at, geo, tweet_id, lang, like_count, quote_count, reply_count, retweet_count, source, text]
        
        # Append the result to the CSV file
        csvWriter.writerow(res)
        counter += 1

    # When done, close the CSV file
    csvFile.close()

    # Print the number of tweets for this iteration
    print('# of Tweets added from this response: ', counter) 


def main(args: argparse.Namespace) -> None:
    if args.test:
        test_request()
        sys.exit(0)
    
    print(f'Keyword: "{args.keyword}"')
    
    print('Prepare headers, dates and output files.')
    bearer_token = auth()
    headers = create_headers(bearer_token)        
    dates = load_dates(args.path_times)
    time_windows = to_time_windows(dates)
    num_time_windows = len(time_windows)
    prepare_output_file(args.path_out)
    total_tweets = 0
    
    print('Start querying.')
    for i, (start_time, end_time) in enumerate(time_windows, start=1):
        if args.skip_first_n:
            if i < args.skip_first_n:
                continue
        start_str = start_time.isoformat(sep='T', timespec='auto') + '.000Z'
        end_str = end_time.isoformat(sep='T', timespec='auto') + '.000Z'
        print(f'Start time: {start_str}, end time. {end_str}, [{i}/{num_time_windows}]')
        count = 0 # Counting tweets per time period
        max_count = args.max_per_tw # Max tweets per time period
        flag = True
        next_token = None
        
        while flag:
            # Check if max_count reached
            if count >= max_count:
                break
            print('-------------------')
            print('Token: ', next_token)
            print(f'Time window: start {start_str}, end {end_str}')
            url = create_url(args.keyword, start_str, end_str, args.max_results)
            json_response = connect_to_endpoint(url[0], headers, url[1])
            result_count = json_response['meta']['result_count']
            
            if 'next_token' in json_response['meta']:
                # Save the token to use for next call
                next_token = json_response['meta']['next_token']
                print('Next Token: ', next_token)
                if result_count is not None and result_count > 0 and next_token is not None:
                    print('Start Date: ', start_time)
                    append_to_csv(json_response, args.path_out)
                    count += result_count
                    total_tweets += result_count
                    print('Total # of Tweets added: ', total_tweets)
                    print('-------------------')
                    time.sleep(5)
                    
            # If no next token exists
            else:
                if result_count is not None and result_count > 0:
                    print('-------------------')
                    print('Start Date: ', start_time)
                    append_to_csv(json_response, args.path_out)
                    count += result_count
                    total_tweets += result_count
                    print('Total # of Tweets added: ', total_tweets)
                    print('-------------------')
                    time.sleep(5)
                
                #Since this is the final request, turn flag to false to move to the next time period.
                flag = False
                next_token = None
            time.sleep(5)
    
    print('Total number of results: ', total_tweets)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_times', help='Path to pickle files containing times (minutes) per month.')
    parser.add_argument('-o', '--path_out', help='Path to output file where tweets are saved (csv).')
    parser.add_argument('-k', '--keyword', default='religion OR spirituality lang:en', help='Twitter keyword.')
    parser.add_argument('-m', '--max_per_tw', type=int, default=1000, 
                        help='Maximum number of tweets to download per time window.')
    parser.add_argument('-M', '--max_results', type=int, default=500, 
                        help='Maximum number of results per query. Twitter max is 500.')
    parser.add_argument('-t', '--test', action='store_true', help='Test run.')
    parser.add_argument('-s', '--skip_first_n', type=int, default=None)
    cmd_args = parser.parse_args()
    main(cmd_args)
