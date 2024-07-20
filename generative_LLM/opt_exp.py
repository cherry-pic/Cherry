from transformers import pipeline, set_seed
from data_loading import load_data
set_seed(32)
generator = pipeline('text-generation', model="facebook/opt-125m")

data = load_data()
prompt1 = "Given the following story STORY: {context}, should this sentence be mentioned in the event coverage? SENTENCE: {sentence} "
prompt2 = "ARTICLE:\n {context} \n\n SENTENCE: {sentence} \n\n QUESTION: Answer with \"yes\" or \"no\" only. Is the above sentence important to mention in a news article that covers the story mentioned in the above article?"
for event_id, event in data.items():
    context = event['documents'][0]
    for sentence in event['sentences']:
        request = prompt2.replace("{context}",context)
        request = request.replace("{sentence}", sentence)
        generated_text = generator(request)
        response = generated_text[0]['generated_text']
