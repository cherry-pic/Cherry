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
    df['fact'] = df['fact'].apply(lambda x: ' '.join([word.lower() for word in word_tokenize(x) ])) #df['fact'].apply(lambda x: ' '.join([word.lower() for word in word_tokenize(x) if not word in (stop)]))
    df['context'] = df['context'].apply(lambda x: ' '.join([word.lower() for word in word_tokenize(x) ])) #df['context'].apply(lambda x: ' '.join([word.lower() for word in word_tokenize(x) if not word in (stop)]))
    return df

# splitting data file into train, val, test
def preprocess_and_split_data(full_ds_file):
    data_splits=[]
    full_dataset_df = pd.read_csv(full_ds_file, delimiter='\t')
    #full_dataset_df = preprocess(full_dataset_df)

    print("Generating data splits ...")

    train_df, test_df = train_test_split(full_dataset_df, test_size=0.15,shuffle=False)  # splitting full ds into 20% testing, and 80% training
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    ds = (train_df, test_df)
    data_splits.append(ds)

    return test_df, train_df



# loads data
def load_data():

    clean_ds_path = "data/clean_data.csv"
    test,train = preprocess_and_split_data(clean_ds_path)
    docs = dict()
    full_json = read_json("data/final_events_7.json")
    full_json = full_json['events']
    for index, row in test.iterrows():  #loading the testing ds
        event_id = row['event_id']
        sentence = row["fact"]
        label = row["label"]
        context = row['context']
        if not event_id in docs:
            docs[event_id]= {"documents":[],"sentences":[], "labels":[]}
        docs[event_id]["sentences"].append(sentence)
        docs[event_id]["labels"].append(label)
        docs[event_id]["context"]=context
        if len(docs[event_id]["documents"]) ==0:
            # get event from file
            for event in full_json:
                if event['id']==event_id:
                    for article in event['neutral_articles']:
                        docs[event_id]["documents"].append(article['text'])
                    for article in event['left_articles']:
                        docs[event_id]["documents"].append(article['text'])
                    for article in event['right_articles']:
                        docs[event_id]["documents"].append(article['text'])
                    break

    return docs

def load_data_with_biased_context():

    clean_ds_path = "data/clean_data.csv"
    test,train = preprocess_and_split_data(clean_ds_path)
    train_docs = dict()
    test_docs = dict()
    full_json = read_json("data/final_events_7.json")
    full_json = full_json['events']
    for index, row in train.iterrows():  #loading the testing ds
        event_id = row['event_id']
        sentence = row["fact"]
        label = row["label"]
        context = row['context']
        if not event_id in train_docs:
            train_docs[event_id]= {"documents":[],"sentences":[], "labels":[]}
        train_docs[event_id]["sentences"].append(sentence)
        train_docs[event_id]["labels"].append(label)
        train_docs[event_id]["context"]=context
        if len(train_docs[event_id]["documents"]) ==0:
            # get event from file
            for event in full_json:
                if event['id']==event_id:
                    #for article in event['neutral_articles']:
                    #    docs[event_id]["documents"].append(article['text'])
                    #for article in event['left_articles']:
                    # adding the first article from each bias direction
                    train_docs[event_id]["documents"].append(event['left_articles'][0]['text'])
                    #for article in event['right_articles']:
                    train_docs[event_id]["documents"].append(event['right_articles'][0]['text'])
                    break
    for index, row in test.iterrows():  #loading the testing ds
        event_id = row['event_id']
        sentence = row["fact"]
        label = row["label"]
        context = row['context']
        if not event_id in test_docs:
            test_docs[event_id]= {"documents":[],"sentences":[], "labels":[]}
        test_docs[event_id]["sentences"].append(sentence)
        test_docs[event_id]["labels"].append(label)
        test_docs[event_id]["context"]=context
        if len(test_docs[event_id]["documents"]) ==0:
            # get event from file
            for event in full_json:
                if event['id']==event_id:
                    #for article in event['neutral_articles']:
                    #    docs[event_id]["documents"].append(article['text'])
                    #for article in event['left_articles']:
                    # adding the first article from each bias direction
                    test_docs[event_id]["documents"].append(event['left_articles'][0]['text'])
                    #for article in event['right_articles']:
                    test_docs[event_id]["documents"].append(event['right_articles'][0]['text'])
                    break

    return train_docs, test_docs