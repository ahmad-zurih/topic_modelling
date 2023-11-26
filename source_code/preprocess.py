import argparse 
from datetime import datetime
import json
import os
import re
from typing import Dict, List, Any 


import emoji
import pandas as pd
from tqdm import tqdm


SENTI_DICT = {
    'NEGATIVE': 'sentinegative',
    'POSITIVE': 'sentipositive'
}


def clean(text: str) -> str:
    ctext = emoji.demojize(text, language='en')
    ctext = re.sub(r':[A-Za-z0-9_]+:', r'', ctext)
    ctext = re.sub(r'&amp;', r'&', ctext)  # handle ampersand encoding errors
    ctext = re.sub(r'^rt @[A-Za-z0-9_]+:', r'', ctext)  # remove retweet leftovers
    ctext = re.sub(r'\S+…', r'', ctext)
    ctext = re.sub(r'http\S+', r'embeddedurl', ctext) # replace by 'embeddedurl'
    ctext = re.sub(r'@', '', ctext)  # remove @ but leave rest of username
    ctext = re.sub(r'#', '', ctext)
    ctext = re.sub(r'\n', ' ', ctext)
    ctext = re.sub(r'\r', ' ', ctext)
    ctext = re.sub(r'\s+', r' ', ctext)
    ctext = re.sub(r'^ ', r'', ctext)
    ctext = re.sub(r' $', '', ctext)
    if '\n' in ctext:
        import pdb; pdb.set_trace()
    return ctext


def process(text: str, sentiment_label: str, sentiment_confidence: float, senti_threshold: float) -> str:
    ltext = text.lower()
    ctext = clean(ltext)
    if sentiment_confidence >= senti_threshold:
        ctext = f'{ctext} {SENTI_DICT[sentiment_label]}'
    return ctext


def get_has_religion(text: str) -> bool:
    return True if 'religion' in text else False


def get_has_spirituality(text: str) -> bool:
    return True if 'spirituality' in text else False


def get_startswith_RT(text: str) -> bool:
    return True if text.startswith('RT') else False


def to_date_time(input_str: str) -> datetime:
    if input_str == 'created_at':
        return None
    return datetime.fromisoformat(input_str)

def get_month(date: datetime) -> int:
    if date:
        return date.month
    return None


def get_cur_df_stats(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        'num_rows': len(df),
        'num_rows_with_spiri': len(df[df.spirituality == True]),
        'num_rows_without_spiri': len(df[df.spirituality == False]),
        'num_rows_with_reli': len(df[df.religion == True]),
        'num_rows_without_reli': len(df[df.religion == False]),
        'num_rows_with_spiri_and_reli': len(df[(df.spirituality == True) & (df.religion == True)]),
        'num_rows_without_spiri_and_reli': len(df[(df.spirituality == False) & (df.religion == False)]),
        'mean_tweet_length': df.tweet.apply(len).mean(),
        'num_retweets': len(df[df.RT == False]),
        'num_embeddedurl': len(df[df.RT == False])
    }


def main(args: argparse.Namespace) -> None:
    stats = {}
    print('Load tweets')
    df = pd.read_csv(args.path_tweets, header=0)
    print('Add date and month columns')
    df['datetime'] = df.apply(lambda x: to_date_time(x['created_at']), axis=1)
    df['month'] = df.apply(lambda x: get_month(x['datetime']), axis=1)
    print('Add keyword-column')
    df['religion'] = df.apply(lambda x: get_has_religion(x.tweet), axis=1)
    df['spirituality'] = df.apply(lambda x: get_has_spirituality(x.tweet), axis=1)
    df['RT'] = df.apply(lambda x: get_startswith_RT(x.tweet), axis=1)
    print('Load sentiments')
    df_sentiment = pd.read_csv(args.path_sentiment, sep='\t', names=['label_name', 'label_score'])
    print('Add sentiments to tweets')
    df = df.assign(sentiment_label=df_sentiment.label_name)
    df = df.assign(sentiment_confidence=df_sentiment.label_score)
    print('Compute initial stats')
    stats['initial_stats'] = get_cur_df_stats(df)
    
    print('Remove duplicates by tweet id')
    orig_len = len(df)
    df = df.drop_duplicates(subset=['id'])
    new_len = len(df)
    print('Compute stats after duplicate removal')
    stats['after_duplicate_removal'] = get_cur_df_stats(df)
    print(f'Number of tweets removed: {orig_len - new_len}')
    
    print('Downsample tweets religion tweets.')
    print(f'Corpus size original: {len(df)}')
    df_spiri = df[df['spirituality'] == True]
    df_reli = df[df['religion'] == True]
    num_spiri = len(df_spiri)
    print('Downsampling.')
    reli_downsampled = df_reli.sample(n=num_spiri, random_state=1)
    df = pd.concat([reli_downsampled, df_spiri])
    print('Compute stats after downsampling')
    stats['after_religion_downsampling'] = get_cur_df_stats(df)
    print('Process')
    with open(os.path.join(args.path_out_dir, 'stats.json'), 'w') as fout:
        json.dump(stats, fout, indent=4)
    df['senti_clean_text'] = df.apply(lambda x: process(x.tweet, x.sentiment_label, x.sentiment_confidence, args.threshold), axis=1)
    
    # Output full dataset
    print('Write full text to csv')
    df.to_csv(args.path_out)
    print('Write full text to txt')
    with open(args.path_raw_out, 'w') as fout:
        for i, row in tqdm(df.iterrows()):
            fout.write(row['senti_clean_text'] + '\n')
    
    # Split dataframe into month-dataframes
    month_dfs = {}
    grouped = df.groupby(df.month)
    for i in range(1, 13):
        df_month = grouped.get_group(i)
        month_dfs[i] = df_month
    
    # Output dataset split by month
    print('Generate monthly datasets')
    for month, month_df in tqdm(month_dfs.items()):
        month_df.to_csv(os.path.join(args.path_out_dir, f'month_{month}.csv'), 'w')
        with open(os.path.join(args.path_out_dir, f'month_{month}_raw.txt'), 'w') as fout:
            for i, row in month_df.iterrows():
                fout.write(row['senti_clean_text'] + '\n')
    
    # Split dataframe into quarter-dataframes
    quarter_dfs = {
        1: pd.concat([grouped.get_group(1), grouped.get_group(2), grouped.get_group(3)]),
        2: pd.concat([grouped.get_group(4), grouped.get_group(5), grouped.get_group(6)]),
        3: pd.concat([grouped.get_group(7), grouped.get_group(8), grouped.get_group(9)]),
        4: pd.concat([grouped.get_group(10), grouped.get_group(11), grouped.get_group(12)])
    }
    
    # Output dataset split by quarters
    print('Generate quarter datasets')
    for quarter, quarter_df in tqdm(quarter_dfs.items()):
        quarter_df.to_csv(os.path.join(args.path_out_dir, f'quarter_{quarter}.csv'))
        with open(os.path.join(args.path_out_dir, f'quarter_{quarter}_raw.txt'), 'w') as fout:
            for i, row in quarter_df.iterrows():
                fout.write(row['senti_clean_text'] + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--path_tweets', help='Path to tweet input file, csv-format.')
    parser.add_argument('-s', '--path_sentiment', help='Path to file containing tweet sentiment, tsv-format.')
    parser.add_argument('-o', '--path_out', help='Path to output file, csv-format.')
    parser.add_argument('-d', '--path_out_dir', help='Path to output directory for monthly and season datasets.')
    parser.add_argument('-r', '--path_raw_out', help='Path to output file containing only cleaned, senti-augmented text.')
    parser.add_argument('--threshold', type=float, default=0.9, help='Set sentiment threshold.')
    cmd_args = parser.parse_args()
    main(cmd_args)
