from transformers import pipeline

llm = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype="auto",
    device=0,                     # 0-cı GPU
)

messages = [
    {"role": "user", "content": "What is RAG in one sentence?"},     # istənilən sual yaz, məsələn: "What is RAG in one sentence?"
]

output = llm(messages, max_new_tokens=100)
print(output)
print(output[0]["generated_text"][-1]["content"])