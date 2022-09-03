from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased-conversational")
model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased-conversational")