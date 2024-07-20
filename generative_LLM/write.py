import codecs


def write_responses(data,exp_dir):
    with codecs.open(exp_dir+"responses.tsv", 'w', encoding='utf8') as out:
        out.write("id\tevent_id\tgold_label\tpredicted_label\tresponse\n")
        for event_id, event in data.items():
            for i, sentence in enumerate(event['sentences']):
                out.write(str(i)+'\t'+event_id+'\t'+str(event['labels'][i])+'\t'+str(event['predicted_labels'][i])+'\t'+event['responses'][i]+'\n')
        out.close()
