from data_loading import load_data
from lexrank import LexRank
from lexrank.mappings.stopwords import STOPWORDS
from torchmetrics import Accuracy,F1Score
import torch
from transformers import AutoTokenizer
import codecs
import datetime
import os
current_time = datetime.datetime.now()
now = current_time.strftime("%d_%m_%Y_%H_%M")
exp_dir = "experiments/exp_" + now+"/"
isExist = os.path.exists(exp_dir)
if not isExist:
    os.makedirs(exp_dir)
from nltk.tokenize import WhitespaceTokenizer
tk = WhitespaceTokenizer()
exp_notes = input('Experiment notes:')

def evaluate(summarizer_sentences, annotated_sentences,labels):

    true_y = labels
    predicted_y = []
    for i, ann_sent in enumerate(annotated_sentences):
        if ann_sent in summarizer_sentences:
            predicted_y.append(1)
        else:
            predicted_y.append(0)

    true_y = torch.tensor(true_y)
    predicted_y = torch.tensor(predicted_y)
    accuracy = Accuracy(average='macro', num_classes=2, task = "binary")
    acc_torch = accuracy(predicted_y, true_y)
    f1_score = F1Score(average='macro', num_classes=2, task = "binary")
    f_torch = f1_score(predicted_y, true_y)


    #f1 = f1_score(y_true, y_pred, average='macro')
    #acc =
    return acc_torch,f_torch


def trim_context(text,max_len):
    trimmed_context = []
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096", truncation_side='right')
    encoded_text = tokenizer(text,padding="max_length")
    token_ids = encoded_text['input_ids'][0:max_len]
    trimmed_text= tokenizer.decode(token_ids)
    trimmed_context.append(trimmed_text)
    return trimmed_context



def trim_by_word(context, context_length):
    words = tk.tokenize(context)
    words = words[0:context_length]  # get the first 100 words from the context only.
    trimmed_context = " ".join(words)  # glue back text
    return trimmed_context



FIXED_CONTEXT = False
SUMM_CONTEXT =  False
SUMM_AND_FIXED = True
CONTEXT_LENGTH = 500  # can be context length (integer) or "all" to consider all document collection as context
if SUMM_CONTEXT or SUMM_AND_FIXED:
    import json
    with open('../experiments/LexRank/data_w_summary/summarized_context_data_500_test.json', 'r') as f:
        docs = json.load(f)
else:
    docs = load_data()

accuracies = []
fs = []
for event_id, event in docs.items():
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

    lxr = LexRank(context, stopwords=STOPWORDS['en'])   # fit the model on document collection

    sentences = event["sentences"]
    labels = event["labels"]

    summary_size = 0
    for label in labels:              # deciding summary size based on the number of positive examples in the dataset for a fair evalutation
        if label == 1:
            summary_size+=1

    if summary_size>0:
        # get summary with classical LexRank algorithm
        summary = lxr.get_summary(sentences, summary_size=summary_size, threshold=.1)
        accuracy,f = evaluate(summary,sentences, labels)
        accuracies.append(accuracy)
        fs.append(f)

        # get summary with continuous LexRank
        summary_cont = lxr.get_summary(sentences, threshold=None)
        print(summary_cont)

final_accuracy = sum(accuracies) / len(accuracies)
final_f = sum(fs)/len(fs)
print("final accuracy = ", final_accuracy)
print("final f score = ", final_f)

with codecs.open(exp_dir + "results.txt", 'w', encoding='utf8') as out:
    out.write(exp_notes + "\n\n")
    out.write("\nTorch metrics' Accuracy = " + str(final_accuracy))
    out.write("\nTorch metrics' F1 = " + str(final_f))
    out.close()