import codecs
import json
import openai
import time
import numpy as np
import os
from dotenv import load_dotenv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

load_dotenv("keys.env")
openai.api_key= os.environ["OPENAI_API_KEY"]

from sentence_transformers import SentenceTransformer,util
lang_model = SentenceTransformer('all-MiniLM-L6-v2')


def call_LLM(prompt):
    RESPONSE_CORRECT = False
    while not RESPONSE_CORRECT:
        try:
            response = openai.ChatCompletion.create(
                model = "gpt-4-turbo",
                messages = [{"role": "user", "content": prompt}],
                temperature = 0,
                #max_tokens = 1,
                top_p = 1,
                frequency_penalty = 0,
                presence_penalty = 0
            )
            text  = response['choices'][0]['message']['content'].strip()
            RESPONSE_CORRECT= True
            #print(text)
        except Exception as e:
            print(e)
            RESPONSE_CORRECT = False
            time.sleep(1)
    return text


def read_json (json_file):
    with open(json_file) as json_file:
        data = json.load(json_file)
        json_file.close()
    return data

def select_events_with_consistent_sources(in_data, out_data, sources): # selects events that have CNN as left-biased, Fox news as right biased and reuters as neutral
    selected_events = []
    data = read_json(in_data)
    events = data['events']
    for event in events:
        left_check = False
        right_check = False
        neutral_check = False
        for article in event['left_articles']:
            if sources['left'] in article['source']:
                left_check = True
        for article in event['right_articles']:
            if sources['right'] in article['source']:
                right_check = True
        for article in event['nuetral_articles']:
            if sources['neutral'] in article['source']:
                neutral_check = True
        if left_check and right_check and neutral_check:
            selected_events.append(event)
    data['events'] = selected_events
    with open(out_data, 'w', encoding='utf-8') as out:
        json.dump(data, out, ensure_ascii=False, indent=4)
        out.close()
    print(f"total number of events selected is {len(selected_events)}")


def calculate_cosine_similarity(text1, text2):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight and efficient model
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)
    similarity = util.cos_sim(embedding1, embedding2)
    return similarity.item()


def calculate_articles_similarities(in_data, out_data): # for each biased article, add its cosine similarity with the first neutral article.
    data = read_json(in_data)
    events = data['events']
    for event in events:
        for a in event['left_articles']:
            sim = calculate_cosine_similarity(a['text'], event['nuetral_articles'][0]['text'])
            a['similarity'] = sim
        print("added left biased articles similarities for event"+event['event_id'])
        for a in event['right_articles']:
            sim = calculate_cosine_similarity(a['text'], event['nuetral_articles'][0]['text'])
            a['similarity'] = sim
        print("added right biased articles similarities for event" + event['event_id'])
    data['events'] = events
    with open(out_data, 'w', encoding='utf-8') as out:
        json.dump(data, out, ensure_ascii=False, indent=4)
        out.close()

def calculate_avg_similarity(in_data):
    sim_sum = 0
    count = 0
    data = read_json(in_data)
    events = data['events']
    for event in events:
        for a in event['left_articles']:
            sim_sum+= a['similarity']
            count+=1
        for a in event['right_articles']:
            sim_sum += a['similarity']
            count += 1
    avg_similarity = sim_sum/count
    print("average similarity is "+str(avg_similarity))


def select_most_biased_articles(in_data, out_data, cos_sim_t):
    selected_events = []
    data = read_json(in_data)
    events = data['events']
    total_biased_articles_collected =0
    for event in events:
        selection = {'biased': [], 'neutral': None}
        for a in event['left_articles']:                # for each left biased article
            if a['similarity'] > cos_sim_t:             # exclude highly similar biased articles
                continue
            if selection['neutral'] is None:        # if the selection of articles for this event is empty
                selection['neutral'] = event['nuetral_articles'][0]    # add first the neutral article (Ground Truth)
            biased_article = {"article": a, "statements":[]}
            other_perspective_sources = [ar['outlet'] for ar in event['left_articles']]
            for cluster_id,st_cluster in event['grouped_statements'].items():
                if st_cluster['importance'] == '0' or st_cluster['importance']=='mixed':  # exclude statements from cluster -1 and statements who are not important.
                    continue
                sources_metnioned_s = [s['outlet'] for s in st_cluster['statements']]   # find all the sources that mentioned the statement in this cluster
                if a['outlet'] in sources_metnioned_s: # exclude statements already mentioned by this biased article.
                    continue
                # check if statementioned mentioned in the other perspective
                mentioned= False
                for s in sources_metnioned_s:
                    if s in other_perspective_sources:
                        mentioned = True
                if not mentioned:
                    continue
                biased_article["statements"].append(st_cluster)   # add important statements missing from this article

            if len(biased_article['statements'])>0:
                selection['biased'].append(biased_article)
        if len(selection['biased']) >0:
            selected_events.append(selection)
        total_biased_articles_collected += len(selection['biased'])
    data['events'] = selected_events
    print("total biased articles collected is "+str(total_biased_articles_collected))
    with open(out_data, 'w', encoding='utf-8') as out:
        json.dump(data, out, ensure_ascii=False, indent=4)
        out.close()


def load_template(template_path):
    with codecs.open(template_path, 'r', encoding='utf8') as f:
        lines= f.readlines()
        template="".join(lines)
        f.close()
    return template
def verify_existence(in_data, out_data):
    data = read_json(in_data)
    events = data['events']
    template = load_template("templates/existence_check.txt")
    for event in events:
        for i, biased_article in enumerate(event['biased']):
            print(f"Verifying existence of importantce statements in article {i}")
            for statement in biased_article['statements']:
                prompt = template.replace("[[statement]]",statement['statements'][0]['text'])
                prompt = prompt.replace("[[news report]]", biased_article['article']['text'])
                response = call_LLM(prompt)
                statement['LLM_response'] = int(response)
    data['events'] = events
    with open(out_data, 'w', encoding='utf-8') as out:
        json.dump(data, out, ensure_ascii=False, indent=4)
        out.close()

def sample_before_after(in_data):
    data = read_json(in_data)
    events = data['events']
    for e, event in enumerate(events):
        for a,biased_article in enumerate(event['biased']):
            biased_article['length'] = len(biased_article['article']['text'])
            sim = biased_article['article']['similarity']
        sorted_data = sorted(event['biased'], key=lambda x: x['length'])

def integrate_missing_statements(in_data, out_data):
    with codecs.open(out_data.replace(".json",'.txt'), 'w', encoding='utf8') as responses_file:
        data = read_json(in_data)
        events = data['events']
        template = load_template("templates/integrate_missing_statement.txt")
        for e, event in enumerate(events):
            print(f"Neutralizing event number {e+1}")
            for a,biased_article in enumerate(event['biased']):
                print(f"Neutralizing biased article number {a+1}")
                updating_article = ""
                i=1
                for statement in biased_article['statements']:
                    if statement['LLM_response'] == 1:                # skipping statements that already exist in the article as indicated by the LLM
                        continue

                    prompt = template.replace("[[statement]]",statement['statements'][0]['text'])
                    if i==1:
                        prompt = prompt.replace("[[news report]]", biased_article['article']['text'])
                    else:
                        prompt = prompt.replace("[[news report]]", updating_article)
                    print(f"integrating statement # {i}")
                    response = call_LLM(prompt)
                    updating_article = response
                    i+=1
                biased_article['balanced_version']= updating_article
                responses_file.write(updating_article+"\n+++++++++++++++++++\n")
        data['events'] = events

        with open(out_data, 'w', encoding='utf-8') as out:
            json.dump(data, out, ensure_ascii=False, indent=4)
            out.close()
        responses_file.close()

def naively_add_missing_statements(in_data, out_data):
    data = read_json(in_data)
    events = data['events']
    for e, event in enumerate(events):
        for a,biased_article in enumerate(event['biased']):
            missing_statements = ""
            for statement in biased_article['statements']:
                if statement['LLM_response'] == 0:                # skipping statements that already exist in the article as indicated by the LLM
                    missing_statements+=statement['statements'][0]['text']+"\n"

            biased_article['balanced_version']= missing_statements+ biased_article['article']['text']
    data['events'] = events

    with open(out_data, 'w', encoding='utf-8') as out:
        json.dump(data, out, ensure_ascii=False, indent=4)
        out.close()


def write_similarities(all_sum_sim_original_neutral,all_sum_sim_balanced_neutral,all_sum_sim_original_balanced):
    with codecs.open("sim_original_neutral.txt", "w", encoding='utf8') as out:
        for sim in all_sum_sim_original_neutral:
            out.write(str(sim))
            out.write("\n")
        out.close()
    with codecs.open("sim_balanced_neutral.txt", "w", encoding='utf8') as out:
        for sim in all_sum_sim_balanced_neutral:
            out.write(str(sim))
            out.write("\n")
        out.close()
    with codecs.open("sim_original_balanced.txt", "w", encoding='utf8') as out:
        for sim in all_sum_sim_original_balanced:
            out.write(str(sim))
            out.write("\n")
        out.close()

def read_similarities_from_files():
    sim_original_balanced =[]
    sim_balanced_neutral = []
    sim_original_neutral =[]
    with codecs.open("sim_original_balanced.txt", "r", encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            sim=float(line)
            sim_original_balanced.append(sim)
        f.close()
    with codecs.open("sim_balanced_neutral.txt", "r", encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            sim=float(line)
            sim_balanced_neutral.append(sim)
        f.close()
    with codecs.open("sim_original_neutral.txt", "r", encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            sim=float(line)
            sim_original_neutral.append(sim)
        f.close()
    return sim_original_balanced, sim_balanced_neutral, sim_original_neutral


def calculate_article_similarity(in_data):
    data = read_json(in_data)
    events = data['events']
    sum_sim_original_neutral = 0
    sum_sim_balanced_neutral = 0
    sum_sim_original_balanced = 0
    counter = 0
    all_sum_sim_original_neutral = []
    all_sum_sim_balanced_neutral = []
    all_sum_sim_original_balanced = []
    for event in events:
        for biased_article in event['biased']:
            original_article = biased_article['article']['text']
            balanced_article = biased_article['balanced_version']
            neutral_article = event['neutral']['text']
            sim_original_neutral = calculate_cosine_similarity(original_article, neutral_article)
            sim_balanced_neutral = calculate_cosine_similarity(balanced_article, neutral_article)
            sim_original_balanced = calculate_cosine_similarity(original_article, balanced_article)
            sum_sim_original_neutral+= sim_original_neutral
            sum_sim_balanced_neutral+= sim_balanced_neutral
            sum_sim_original_balanced+= sim_original_balanced
            all_sum_sim_original_neutral.append(sim_original_neutral)
            all_sum_sim_balanced_neutral.append(sim_balanced_neutral)
            all_sum_sim_original_balanced.append(sim_original_balanced)
            counter+=1
    avg_sim_original_neutral = sum_sim_original_neutral/counter
    avg_sim_original_balanced = sum_sim_original_balanced/counter
    avg_sim_balanced_neutral = sum_sim_balanced_neutral/counter

    print(f"Average similarity between original articles and neutral articles (GT) = {avg_sim_original_neutral}")
    print(f"Average similarity between original articles and balanced articles = {avg_sim_original_balanced}")
    print(f"Average similarity between balanced articles and neutral articles = {avg_sim_balanced_neutral}")
    write_similarities(all_sum_sim_original_neutral,all_sum_sim_balanced_neutral,all_sum_sim_original_balanced)


def calculate_avg_article_similarity(sim_original_balanced, sim_balanced_neutral, sim_original_neutral):
    print(f"Average similarity between original articles and neutral articles (GT) = {sum(sim_original_neutral)/len(sim_original_neutral)}")
    print(f"Average similarity between original articles and balanced articles = {sum(sim_original_balanced)/len(sim_original_balanced)}")
    print(f"Average similarity between balanced articles and neutral articles = {sum(sim_balanced_neutral)/len(sim_balanced_neutral)}")

def calculate_distances(similarities):
    distances =[]
    for sim in similarities:
        d_f = 1-sim
        d_r = round(d_f, 4)
        #d = d_r*100
        distances.append(d_r)
    return distances
def visualize_similarities(sim_original_neutral,sim_balanced_neutral):
    dist_original_neutral = calculate_distances(sim_original_neutral)
    dist_balanced_neutral = calculate_distances(sim_balanced_neutral)

    y_values = list(range(len(dist_original_neutral)))

    # Plot the two distance sets with different markers.
    plt.scatter(dist_original_neutral, y_values, label="List 1", color="blue", marker="o")
    plt.scatter(dist_balanced_neutral, y_values, label="List 2", color="red", marker="x")

    # Add labels, title, and legend
    plt.xlabel("Cosine distance")
    plt.ylabel("")
    plt.title("Cosine distance comparison.")
    plt.legend()
    plt.grid(True)

    plt.show()




def get_before_after_arrangement(distance_values):
    couples = []
    for i in range(len(distance_values)):
        couple = (distance_values[i],(i/100))
        couples.append(couple)
    return couples
def visualize_shifting(sim_original_neutral,sim_balanced_neutral):
    # Example data points
    dist_original_neutral = calculate_distances(sim_original_neutral)
    dist_balanced_neutral = calculate_distances(sim_balanced_neutral)
    before = get_before_after_arrangement(dist_original_neutral)
    after = get_before_after_arrangement(dist_balanced_neutral)

    original_points = np.array(before)
    translated_points = np.array(after)

    #original_points = [(1, 2), (3, 4), (5, 6), (7, 8)]
    # Convert points and vector to numpy arrays
    #original_points = np.array(original_points)
    #translation_vector = np.array(translation_vector)

    # Calculate translated points
    #translated_points = original_points + translation_vector

    # Create a new plot
    plt.figure(figsize=(8, 8))

    # Plot the original and translated points
    plt.scatter(original_points[:, 0], original_points[:, 1], edgecolor='gray', facecolor='none', s=100, label='Pre-correction')
    plt.scatter(translated_points[:, 0], translated_points[:, 1], marker='x', color='black', s=100, label='Post-correction')

    # Draw lines connecting each original point to its translated counterpart
    for orig, trans in zip(original_points, translated_points):
        plt.plot([orig[0], trans[0]], [orig[1], trans[1]], color='gray', linestyle='--')

    # Add labels and legend
    #plt.title("Cosine distance from neutral news reports pre and post correction.", fontsize=16, fontweight='bold')
    plt.xlabel("Cosine distance", fontsize=20) #, fontweight='bold'
    plt.ylabel("")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=20)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().yaxis.set_ticks([])
    plt.xticks(fontsize=20)
    plt.text(-0.02, 0.0, "Neutral news report", fontsize=20,  color='black', rotation='vertical') # fontweight='bold',
    # Show the plot
    plt.show()

def get_before_after_arrangement2(distance_values):
    couples = []
    for i in range(len(distance_values)):
        couple = ((i/100), distance_values[i])
        couples.append(couple)
    return couples
def visualize_shifting2(sim_original_neutral,sim_balanced_neutral):
    # Example data points
    dist_original_neutral = calculate_distances(sim_original_neutral)
    dist_balanced_neutral = calculate_distances(sim_balanced_neutral)
    before = get_before_after_arrangement2(dist_original_neutral)
    after = get_before_after_arrangement2(dist_balanced_neutral)

    original_points = np.array(before)
    translated_points = np.array(after)

    #original_points = [(1, 2), (3, 4), (5, 6), (7, 8)]
    # Convert points and vector to numpy arrays
    #original_points = np.array(original_points)
    #translation_vector = np.array(translation_vector)

    # Calculate translated points
    #translated_points = original_points + translation_vector

    # Create a new plot
    plt.figure(figsize=(10, 8))

    # Plot the original and translated points
    plt.scatter(original_points[:, 0], original_points[:, 1], edgecolor='gray', facecolor='none', s=100, label='Pre-correction')
    plt.scatter(translated_points[:, 0], translated_points[:, 1], marker='x', color='black', s=100, label='Post-correction')

    # Draw lines connecting each original point to its translated counterpart
    for orig, trans in zip(original_points, translated_points):
        plt.plot([orig[0], trans[0]], [orig[1], trans[1]], color='gray', linestyle='--')

    # Add labels and legend
    #plt.title("Cosine distance from neutral news reports pre and post correction.", fontsize=16, fontweight='bold')
    plt.ylabel("Cosine distance", fontsize=20) #, fontweight='bold'
    plt.xlabel("")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=20)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().xaxis.set_ticks([])
    plt.yticks(fontsize=20)
    plt.text(0.0, 0.0, "Neutral news report", fontsize=20,  color='black') # fontweight='bold', rotation='vertical'
    # Show the plot
    plt.show()


_1_raw_data_path = "bias_analysis_events_clustered_wpredictions.json"
_2_baseline_data_path = "baseline_data.json"
_3_baseline_data_path_w_sim = "baseline_data_w_sim.json"
_4_ready_data_path = "ready_data.json"
_5_existence_checked_data_path =  "existence_checked_data.json"
_6_raw_generated_reports_data_path = "raw_generated_reports_data.json"
_7_generated_reports_data_path = "generated_reports_data.json"


# select_events_with_consistent_sources(_1_raw_data_path, _2_baseline_data_path, {"left":"cnn.com", "right":"foxnews.com","neutral":"reuters.com"})
# calculate_articles_similarities(_2_baseline_data_path, _3_baseline_data_path_w_sim)
# calculate_avg_similarity(_3_baseline_data_path_w_sim)
# select_most_biased_articles(_3_baseline_data_path_w_sim, _4_ready_data_path, cos_sim_t=0.7)
# verify_existence(_4_ready_data_path, _5_existence_checked_data_path)
# naively_add_missing_statements(_5_existence_checked_data_path, _6_raw_generated_reports_data_path)
# calculate_article_similarity(_6_raw_generated_reports_data_path)

# sim_original_balanced, sim_balanced_neutral, sim_original_neutral = read_similarities_from_files()
# calculate_avg_article_similarity(sim_original_balanced, sim_balanced_neutral, sim_original_neutral)
# visualize_shifting(sim_original_neutral,sim_balanced_neutral)
# visualize_similarities(sim_original_neutral,sim_balanced_neutral)
# integrate_missing_statements(_5_existence_checked_data_path, _7_generated_reports_data_path)
# sample_before_after(_5_existence_checked_data_path)
sim_original_balanced, sim_balanced_neutral, sim_original_neutral = read_similarities_from_files()
visualize_shifting2(sim_original_neutral,sim_balanced_neutral)