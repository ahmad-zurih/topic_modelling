# Execution instructions

```bash
python3 generate_random_dates_per_month.py -n 1440 -y 2021 
```


```bash
python3 download_tweets.py -p /home/user/jgoldz/reli-spiri/chosen_time_windows.pickle -o /srv/scratch0/jgoldz/reli-spiri/downloaded_tweets_religion_hours.csv -k "religion lang:en"
```

```bash
python3 download_tweets.py -p /home/user/jgoldz/reli-spiri/chosen_time_windows.pickle -o /srv/scratch0/jgoldz/reli-spiri/downloaded_tweets_spirituality_hours.csv -k "spirituality lang:en" 
```

```bash
python3 predict_tweet_sentiment.py -i /srv/scratch0/jgoldz/reli-spiri/downloaded_tweets_religion_spirituality_hours.csv -o /srv/scratch0/jgoldz/reli-spiri/reli_spiri_hours_sentiment_raw_results.tsv -c -1 -g 7
```

```bash
python3 preprocess.py -t /srv/scratch0/jgoldz/reli-spiri/downloaded_tweets_religion_spirituality_hours.csv -s /srv/scratch0/jgoldz/reli-spiri/reli_spiri_hours_sentiment_raw_results.tsv -o /srv/scratch0/jgoldz/reli-spiri/religion_spirituality_hours_processed.csv -d /srv/scratch0/jgoldz/reli-spiri/ -r /srv/scratch0/jgoldz/reli-spiri/religion_spirituality_hours_processed_text_only.txt
```
