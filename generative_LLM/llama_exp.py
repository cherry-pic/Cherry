from transformers import AutoTokenizer
import transformers
import torch
from data_loading import load_data
model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline("text-generation", model=model,torch_dtype=torch.float16,device_map="auto",)
data = load_data()
template = "ARTICLE:\n [[context]] \n\n SENTENCE: [[sentence]] \n\n QUESTION: Answer with \"yes\" or \"no\" only. Is the above sentence important to mention in a news article that covers the story mentioned in the above article?"
for event_id, event in data.items():
    context = event['documents'][0]['text']
    for sentence in event['sentences']:
        prompt = template.replace("[[context]]",context)
        prompt = prompt.replace("[[sentence]]", sentence)
        sequences = pipeline(
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=200,
        )
        for seq in sequences:
            print(f"Result: {seq['generated_text']}")