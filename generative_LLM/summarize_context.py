import codecs

import openai
import time
import os
openai.api_key="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"
from data_loading import load_data
import json
import tiktoken

exp_dir = "data_w_summary/"
isExist = os.path.exists(exp_dir)
if not isExist:
    os.makedirs(exp_dir)


# truncates requestes longer than 16k tokens
def truncate(string: str, encoding_name: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    encoded = encoding.encode(string)
    if len(encoded)>=16000:
        encoded = encoded[0:16000]
    truncated = encoding.decode(encoded)
    return truncated


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


data = load_data()

i=0
for event_id, event in data.items():
    print(event_id)
    data[event_id]['summarized_context']=""
    prompt="Summarize the following news articles in one short articles of less than 260 words."
    for doc in data[event_id]['documents']:
        prompt+="\n\n Article #"+str(i)+doc
    prompt+="\n"
    truncated_prompt = truncate(prompt,'gpt-3.5-turbo-16k')
    response = submit_request(truncated_prompt)
    data[event_id]['summarized_context']=response
    i += 1

json_object = json.dumps(data, indent=4)
with open(exp_dir+"summarized_context_data_test.json", "w") as outfile:
    outfile.write(json_object)

with codecs.open("clean_data_w_summary_test.csv", "w", encoding="utf8") as out:
    out.write("event_id\tfactlets_cluster_id\tfact\tsummarized_context\tlabel")
    for event_id, event in data.items():
        summarized_context = data[event_id]['summarized_context']
        summarized_context = summarized_context.replace('\t', '')
        summarized_context = summarized_context.replace('\n', ' ')
        summarized_context = summarized_context.replace('\r', ' ')
        summarized_context = summarized_context.replace("\"", "“")
        summarized_context = summarized_context.strip()
        context = data[event_id]['context']
        facts = data[event_id]["sentences"]
        for i,fact in enumerate(facts):
            label = data[event_id]["labels"][i]
            out.write(event_id+'\t0\t'+fact+'\t'+summarized_context+'\t'+str(label)+'\n')
    out.close()

