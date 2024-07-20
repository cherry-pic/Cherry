import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk import word_tokenize
stop = stopwords.words('english')
import json

def read_json (json_file):
    with open(json_file) as json_file:
        data = json.load(json_file)
        json_file.close()
    return data

def preprocess(df):
    print("Preprocessing data ...")
    df['fact'] = df['fact'].apply(lambda x: ' '.join([word.lower() for word in word_tokenize(x) if not word in (stop)]))
    df['context'] = df['context'] #.apply(lambda x: ' '.join([word.lower() for word in word_tokenize(x) if not word in (stop)]))
    return df


def preprocess_article(text):
    preprocessed = ' '.join([word.lower() for word in word_tokenize(text) if not word in (stop)])
    return preprocessed

# splitting data file into train, val, test
def preprocess_and_split_data(full_ds_file):
    data_splits=[]
    full_dataset_df = pd.read_csv(full_ds_file, delimiter='\t')
    full_dataset_df = preprocess(full_dataset_df)

    print("Generating data splits ...")

    train_df, test_df = train_test_split(full_dataset_df, test_size=0.15,shuffle=False)  # splitting full ds into 20% testing, and 80% training
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    ds = (train_df, test_df)
    data_splits.append(ds)

    return test_df, data_splits



# loads data
def load_data():

    clean_ds_path = "data/clean_data.csv"
    test,data_splits = preprocess_and_split_data(clean_ds_path)
    docs = dict()
    full_json = read_json("data/final_events_7.json")
    full_json = full_json['events']
    for index, row in test.iterrows():
        event_id = row['event_id']
        sentence = row["fact"]
        label = row["label"]
        context = row['context']
        if not event_id in docs:
            docs[event_id]= {"documents":[],"sentences":[], "labels":[]}
        docs[event_id]["sentences"].append(sentence)
        docs[event_id]["labels"].append(label)
        docs[event_id]["context"] = context
        if len(docs[event_id]["documents"]) ==0:
            # get event from file
            for event in full_json:
                if event['id']==event_id:
                    for article in event['neutral_articles']:
                        #article = preprocess_article(article['text'])
                        docs[event_id]["documents"].append(article['text'])
                    for article in event['left_articles']:
                        #article = preprocess_article(article['text'])
                        docs[event_id]["documents"].append(article['text'])
                    for article in event['right_articles']:
                        #article = preprocess_article(article['text'])
                        docs[event_id]["documents"].append(article['text'])
                    break

    return docs

