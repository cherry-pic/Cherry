import codecs
import os
from nltk.tokenize import WhitespaceTokenizer
tk = WhitespaceTokenizer()


exp_dir = "data_w_fixed_context/"
isExist = os.path.exists(exp_dir)
if not isExist:
    os.makedirs(exp_dir)



def trim_context(text, trim):
    words = tk.tokenize(text)
    words = words[0:trim]  #get the first 100 words from the context only.
    trimmed_text = " ".join(words)  # glue back text
    return trimmed_text


TRIM= 500
DS = "clean_data_w_summary_500_train.csv"
with codecs.open(DS, 'r', encoding='utf-8') as f:
    with codecs.open("data_w_fixed_context/"+DS.replace(".csv","")+"_fixed_"+str(TRIM)+".csv", "w", encoding="utf8") as out:
        out.write("event_id\tfactlets_cluster_id\tfact\tcontext\tlabel\n")
        f.readline()
        lines=f.readlines()
        for line in lines:
            event_id,factlets_cluster_id,fact,context,label = line.strip().split('\t')
            fixed_context = trim_context(context, TRIM)
            out.write(event_id+'\t'+factlets_cluster_id+'\t'+fact+'\t'+fixed_context+'\t'+str(label)+'\n')
    out.close()

