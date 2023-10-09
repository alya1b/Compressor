import torch
from transformers import AlbertTokenizer, GPT2LMHeadModel

tokenizer = AlbertTokenizer.from_pretrained("Tereveni-AI/gpt2-124M-uk-fiction")
model = GPT2LMHeadModel.from_pretrained("Tereveni-AI/gpt2-124M-uk-fiction")

input_text = "Но зла Юнона, суча дочка, не"
input_ids = tokenizer.encode(input_text, add_special_tokens=False, return_tensors='pt')

# Generate probabilities for the next word
with torch.no_grad():
    logits = model(input_ids).logits

# Get the last token's probabilities
next_word_logits = logits[:, -1, :]

# Get the 10 words with the highest probabilities
top_k = 10
top_next_word_indices = torch.topk(next_word_logits, k=top_k, dim=-1).indices[0]
top_next_words = [tokenizer.decode(idx.item()) for idx in top_next_word_indices]

print(f"The 10 words with the highest probabilities to be next: {top_next_words}")
