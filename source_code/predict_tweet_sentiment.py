import argparse
import csv
from tqdm import tqdm as tqdm
from transformers import pipeline


# path_p11 = '/srv/scratch0/jgoldz/P11/'
# path_tweets = os.path.join(path_p11, '8set_ALL.name_text_source_ASCII_cleaned.txt')
# path_sentiment_raw = os.path.join(path_p11, 'sentiment_raw_results.txt')
# path_tweets_sentiment = os.path.join(path_p11, '8set_ALL.name_text_source_ASCII_cleaned_w_sentiment.txt')


def main(args: argparse.Namespace) -> None:
    sentiment_analysis = pipeline('sentiment-analysis', model='siebert/sentiment-roberta-large-english', device=args.gpu)

    # Load tweets
    print('Load tweets.')
    rows = []
    with open(args.path_in) as fin:
        reader = csv.reader(fin, delimiter='\t')
        for row in reader:
            rows.append(row)

    print('Start prediction.')
    with open(args.path_in) as fin, open(args.path_out, 'w') as fout:
        reader = csv.reader(fin)
        for row in tqdm(reader):
            sentiment = sentiment_analysis(row[args.column])
            fout.write(f"{sentiment[0]['label']}\t{sentiment[0]['score']}\n")
    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--path_in', help='Path to input file, csv-format.')
    parser.add_argument('-c', '--column', type=int, help='Column index of column that contains tweet text.')
    parser.add_argument('-o', '--path_out', help='Path to output file.')
    parser.add_argument('-g', '--gpu', type=int, help='GPU num to use.')
    cmd_args = parser.parse_args()
    main(cmd_args)
