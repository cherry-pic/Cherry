import codecs
from statistics import mean
import openai
import time
import os
openai.api_key=os.environ["OPENAI_API_KEY"]
from data_loading import load_data,load_data_with_biased_context
import json
import tiktoken

exp_dir = "../experiments/LexRank/data_w_summary/"
isExist = os.path.exists(exp_dir)
if not isExist:
    os.makedirs(exp_dir)


# truncates requestes longer than 16k tokens
def truncate(string: str, encoding_name: str) -> int:
    truncate_at = 16000
    encoding = tiktoken.encoding_for_model(encoding_name)
    encoded = encoding.encode(string)
    if len(encoded)>=truncate_at:
        encoded = encoded[0:truncate_at]
    truncated = encoding.decode(encoded)

    return truncated



def sum_context_length():
    import json
    all_summ_contexts = []
    with open("../experiments/LexRank/data_w_summary/summarized_context_data_test.json") as json_file:
        test_data = json.load(json_file)
        for event_id, event in test_data.items():
            all_summ_contexts.append(event["summarized_context"])
        json_file.close()
    with open("../experiments/LexRank/data_w_summary/summarized_context_data_train.json") as json_file:
        train_data = json.load(json_file)
        for event_id, event in train_data.items():
            all_summ_contexts.append(event["summarized_context"])
        json_file.close()

    num_tokens = []
    for context in all_summ_contexts:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-16k")
        encoded = encoding.encode(context)
        num_tokens.append(len(encoded))
    print("Average number of tokens = "+str(mean(num_tokens)))



def submit_request(prompt):
    RESPONSE_CORRECT = False
    while not RESPONSE_CORRECT:
        try:
            response = openai.ChatCompletion.create(
                model = "gpt-3.5-turbo-16k",
                messages = [{"role": "user", "content": prompt}],
                temperature = 0,
                max_tokens = 250,
                top_p = 1,
                frequency_penalty = 0,
                presence_penalty = 0
            )
            text  = response['choices'][0]['message']['content'].strip()
            #usage = response['usage']['total_tokens']
            RESPONSE_CORRECT= True
            print(text)
        except Exception as e:
            print(e)
            RESPONSE_CORRECT = False
            time.sleep(5)
    return text

def summarize(data,length):
    i=0
    for event_id, event in data.items():
        print(event_id)
        data[event_id]['summarized_context']=""
        prompt="Summarize the following news articles in one short article of "+str(length)+" words."
        for doc in data[event_id]['documents']:
            prompt+="\n\n Article #"+str(i)+doc
        prompt+="\n"
        truncated_prompt = truncate(prompt,'gpt-3.5-turbo-16k')
        response = submit_request(truncated_prompt)
        data[event_id]['summarized_context']=response
        i += 1
    return data

def write_summarized_data(summ_data,data_name,length):
    json_object = json.dumps(summ_data, indent=4)
    with open(exp_dir + "summarized_context_data_"+str(length)+"_"+data_name+".json", "w") as outfile:
        outfile.write(json_object)

    with codecs.open("clean_data_w_summary_"+str(length)+"_"+data_name+".csv", "w", encoding="utf8") as out:
        out.write("event_id\tfactlets_cluster_id\tfact\tcontext\tlabel\n")
        for event_id, event in summ_data.items():
            summarized_context = summ_data[event_id]['summarized_context']
            summarized_context = summarized_context.replace('\t', '')
            summarized_context = summarized_context.replace('\n', ' ')
            summarized_context = summarized_context.replace('\r', ' ')
            summarized_context = summarized_context.replace("\"", "“")
            summarized_context = summarized_context.strip()
            context = summ_data[event_id]['context']
            facts = summ_data[event_id]["sentences"]
            for i, fact in enumerate(facts):
                label = summ_data[event_id]["labels"][i]
                out.write(event_id + '\t0\t' + fact + '\t' + summarized_context + '\t' + str(label) + '\n')
        out.close()
def main(train, test, length):
    sum_context_length()
    train_data,test_data = load_data_with_biased_context()
    if train:
        summarized_train = summarize(train_data,length)
        write_summarized_data(summarized_train, "train",length)
    if test:
        summarized_test = summarize(test_data,length)
        write_summarized_data(summarized_test, "test",length)



main(train=True, test=True, length = 500)
