from transformers import AutoTokenizer

# 加载对应的 tokenizer
tokenizer = AutoTokenizer.from_pretrained('/mnt/data/models/Dynamic_moe')

# 打印 eos token 和其对应的 ID
eos_token = tokenizer.bos_token
eos_token_id = tokenizer.bos_token_id
print(f"EOS Token: {eos_token}, EOS Token ID: {eos_token_id}")