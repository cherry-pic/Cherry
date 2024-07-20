import tiktoken
from data_loading import load_data
from statistics import mean
from sklearn.metrics import confusion_matrix
import codecs

def truncate(string, encoding_name, length) :
    encoding = tiktoken.encoding_for_model(encoding_name)
    encoded = encoding.encode(string)
    if len(encoded) >= length:
        encoded = encoded[0:length]
    truncated = encoding.decode(encoded)

    return truncated

def truncate_by_words(text,length):
    words= text.split(" ")
    if len(words) >= length:
        words = words[0:length]  #get the first 100 words from the context only.
    trimmed_text = " ".join(words)  # glue back text
    return trimmed_text
def get_confusion_matrix(results_file):
    with codecs.open(results_file,'r',encoding='utf8') as f:
        y_true = []
        y_pred = []
        f.readline()
        lines=f.readlines()
        for line in lines:
            fields = line.split('\t')
            y_true.append(int(fields[2]))
            y_pred.append(int(fields[3]))
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print("(tn, fp, fn, tp)")
    print((tn, fp, fn, tp))

# print("128 context:\n")
# get_confusion_matrix("responses_128.tsv")
# print("summarized context:\n")
# get_confusion_matrix("responses_summ.tsv")

contexts = load_data()

lengths = []
for context in contexts:
    truncated_context =  truncate_by_words(context, 400) # (context, "gpt-3.5-turbo-16k", 512)
    paragraphs = truncated_context.split('\n')
    lengths.append(len(paragraphs))
    print(truncated_context)
    print("------------------")
print(mean(lengths))
print(lengths)