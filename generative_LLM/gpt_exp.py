import openai
import time
import os
#
openai.api_key= os.environ["OPENAI_API_KEY"]
from data_loading import load_data
from transformers import AutoTokenizer
from response_parser import parse_response
from write import write_responses
from evaluate import *
import datetime
current_time = datetime.datetime.now()
now = current_time.strftime("%d_%m_%Y_%H_%M")
exp_dir = "experiments/exp_" + now+"/"
isExist = os.path.exists(exp_dir)
if not isExist:
    os.makedirs(exp_dir)
from nltk.tokenize import WhitespaceTokenizer
tk = WhitespaceTokenizer()
exp_notes = input('Experiment notes:')


def load_template(template_id):
    with codecs.open("templates/template_"+str(template_id)+".txt", 'r', encoding='utf8') as f:
        lines= f.readlines()
        template="".join(lines)
        f.close()
    return template

def trim_context(text,max_len):
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096", truncation_side='right')
    encoded_text = tokenizer(text,padding="max_length")
    token_ids = encoded_text['input_ids'][0:max_len]
    trimmed_text= tokenizer.decode(token_ids)
    trimmed_text= trimmed_text.replace("</s>","")
    trimmed_text = trimmed_text.replace("<s>", "")
    trimmed_text= trimmed_text.replace("<pad>", "")
    return trimmed_text

def trim_by_word(context, context_length):
    words = tk.tokenize(context)
    words = words[0:context_length]  # get the first 100 words from the context only.
    trimmed_context = " ".join(words)  # glue back text
    return trimmed_context
def submit_request(prompt):
    RESPONSE_CORRECT = False
    while not RESPONSE_CORRECT:
        try:
            response = openai.ChatCompletion.create(
                model = "gpt-3.5-turbo-16k",
                messages = [{"role": "user", "content": prompt}],
                temperature = 0,
                max_tokens = 2,
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


CONTEXT_LENGTH = 500
SUMM_CONTEXT = False
FIXED_CONTEXT = False
SUMM_AND_FIXED = True
template = load_template(3)  # best 0-shot templete=0, best 10 shot template =3
if SUMM_CONTEXT or SUMM_AND_FIXED:
    import json
    with open('../experiments/LexRank/data_w_summary/summarized_context_data_500_test.json', 'r') as f:
        data = json.load(f)
else:
    data = load_data()

gold_labels = []
predicted_labels = []

for event_id, event in data.items():
    print(event_id)
    data[event_id]['responses'] = []
    data[event_id]['predicted_labels']=[]
    if SUMM_CONTEXT:
        context = event['summarized_context' ]
    elif FIXED_CONTEXT:
        context = event['context']
        context = trim_by_word(context,CONTEXT_LENGTH)
    elif SUMM_AND_FIXED:
        context = event['summarized_context']
        context = trim_by_word(context, CONTEXT_LENGTH)
    else:
        context = event['context']
        context = trim_context(context,CONTEXT_LENGTH)
    for i,sentence in enumerate(event['sentences']):
        print("sentence: "+str(i))
        gold_label = event['labels'][i]
        prompt = template.replace("[[context]]", context)
        prompt = prompt.replace("[[sentence]]", sentence)
        response = submit_request(prompt)
        predicted_label = parse_response(response,gold_label)
        data[event_id]['responses'].append(response)
        data[event_id]['predicted_labels'].append(predicted_label)
        gold_labels.append(gold_label)
        predicted_labels.append(predicted_label)


evaluate(gold_labels,predicted_labels,exp_dir,exp_notes)
write_responses(data,exp_dir)




