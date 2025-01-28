# STEP #5: this script spots cherry-picking by finding important statements that are not mentioned
# in biased articles and generates a score for each article, then combines articles scores per source/outlet.
from sentence_transformers import SentenceTransformer
lang_model = SentenceTransformer('all-MiniLM-L6-v2')
import codecs
from numpy import array, average
from nltk import sent_tokenize
from scipy import spatial

import string
translation_table = str.maketrans('', '', string.digits)
import json
def read_json (json_file):
    with open(json_file) as json_file:
        data = json.load(json_file)
        json_file.close()
    return data

def get_cluster_centroid_vector(cluster):
    vectors = []
    for statement in cluster:
        vectors.append(lang_model.encode(statement['text']))
    vecs = array(vectors)
    centroid = average(vecs, axis=0)
    return centroid

def calculate_min_distance_between_article_and_vector(vector,articles):
    distances = []
    statements = []
    for article in articles:
        sentences = sent_tokenize(article)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 10:   # exclude sentences whose length is less than 10 characters
                sent_vector = lang_model.encode(sent)
                distance = 1 - spatial.distance.cosine(sent_vector, vector)
                distances.append(distance)
                statements.append(sent)

    index_of_min = distances.index(min(distances))
    most_similar_statement = statements[index_of_min]
    return min(distances),most_similar_statement

def exclude_close_statements_from_missing_using_centroids(missing_statements,events, threshold):
    for event_id, missings in missing_statements.items():
        print("Excluding similar statements for event : "+event_id)
        for outlet, missing_states in missings.items():
            new_missing_states = []
            for s in missing_states:
                missing_cluster_id = s[1]  # get the ID of the missing cluster
                for event in events:       # get the whole cluster
                    if event["event_id"] == event_id:   # get the event using its id to extract the whole articles in the event
                        cluster= event["grouped_statements"][missing_cluster_id]
                        cluster_centroid = get_cluster_centroid_vector(cluster["statements"])  # get the vector of the centroid of this cluster
                        #cluster_centroid = lang_model.encode(cluster["statements"][0]["text"])
                        outlet_articles = []                             # get all the articles from this outlet that covered this event
                        for article in event["left_articles"]:
                            if article["outlet"] ==outlet:
                                outlet_articles.append(article["text"])
                        for article in event["right_articles"]:
                            if article["outlet"] ==outlet:
                                outlet_articles.append(article["text"])
                        for article in event["nuetral_articles"]:
                            if article["outlet"] ==outlet:
                                outlet_articles.append(article["text"])

                        min_distance, most_similar_sent = calculate_min_distance_between_article_and_vector(cluster_centroid, outlet_articles)  # calculate the min distance between the centroid of the cluster and all the statements in the articles covered by this outlet
                        if min_distance > threshold:                                                                         # if the min distance is larger than tthe threshold, then there is no single sentence in the article close to the missing statements in the cluster
                            new_missing_states.append(tuple((cluster["statements"][0]["text"],missing_cluster_id)))          # in this case, add this as a certainly missing statement
                        #else:
                            #print("Cluster statement:", cluster["statements"][0]['text'])
                            #print("Most similar sentence: ", most_similar_sent)

            missing_statements[event_id][outlet]= new_missing_states
    return missing_statements


def spot_missing_statements(events_with_predictions_json,distance_threshold,exclude_close_statements):
    data = read_json(events_with_predictions_json)
    events  = data['events']
    missing_statements = dict()
    for event in events:
        event_covering_sources = dict()
        for article in event['left_articles']:   # getting all sources that cover the event
            if not article['outlet'] in event_covering_sources:
                event_covering_sources[article['outlet']]=[]
        for article in event['right_articles']:
            if not article['outlet'] in event_covering_sources:
                event_covering_sources[article['outlet']]=[]
        for article in event['nuetral_articles']:
            if not article['outlet'] in event_covering_sources:
                event_covering_sources[article['outlet']] = []

        for cluster_id, cluster in event["grouped_statements"].items():
            if cluster["importance"]=="1" and int(cluster_id)>-1:  # if the statement is important and does not belong to cluster -1
                statement_covering_sources=[]
                for statement in cluster["statements"]:
                    if not statement["outlet"] in statement_covering_sources:
                        statement_covering_sources.append(statement["outlet"])
                for source, missing_stat in event_covering_sources.items():  # check for the existence of every outlet in the cluster, if it does not exist, add the missing statement to the outlet in the dictionary
                    if not source in statement_covering_sources:
                        event_covering_sources[source].append(tuple((cluster["statements"][0]["text"],cluster_id)))

        missing_statements[event["event_id"]]=event_covering_sources

    if exclude_close_statements:
        missing_statements = exclude_close_statements_from_missing_using_centroids(missing_statements, events, threshold=distance_threshold)

    missing_avg_by_outlet = dict()    # calculate average number of missing important statements per article
    for event_id, covering_outlets in missing_statements.items():
        for outlet, missings in covering_outlets.items():
            if not outlet in missing_avg_by_outlet:
                missing_avg_by_outlet[outlet] = dict()
                missing_avg_by_outlet[outlet]["missing_sum"]=0
                missing_avg_by_outlet[outlet]["total_events"] = 0
                missing_avg_by_outlet[outlet]["missing_avg"] = 0
            missing_avg_by_outlet[outlet]["missing_sum"] +=len(missings)
            missing_avg_by_outlet[outlet]["total_events"] +=1
            missing_avg_by_outlet[outlet]["missing_avg"] = missing_avg_by_outlet[outlet]["missing_sum"]/missing_avg_by_outlet[outlet]["total_events"]

    print(missing_avg_by_outlet)

    return missing_statements, missing_avg_by_outlet




