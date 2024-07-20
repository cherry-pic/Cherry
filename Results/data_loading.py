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
    contexts = []
    ids = []
    clean_ds_path = "clean_data.csv"
    test,train = preprocess_and_split_data(clean_ds_path)
    docs = dict()
    full_json = read_json("final_events_7.json")
    full_json = full_json['events']
    for index, row in test.iterrows():  #loading the testing ds
        event_id = row['event_id']
        # get event from file
        for event in full_json:
            if event['id']==event_id:
                if not event_id in ids:
                    contexts.append(event['neutral_articles'][0]['text'])
                    ids.append(event_id)
                break

    return contexts

