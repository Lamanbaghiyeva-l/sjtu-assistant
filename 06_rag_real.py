import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import json

# ---------- 1. MODELLƏR ----------
embed_model = SentenceTransformer("/mnt/pool1/sharehome/laman/workspace/resume_screening/models/all-MiniLM-L6-v2")

llm = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype="auto",
    device=0,
)

# ---------- 2. SƏNƏDLƏR ----------
chunks = json.load(open("data/chunks.json", encoding="utf-8"))


# ---------- 3. SUAL ----------
question = "When does the library close?"
THRESHOLD = 0.3

# ---------- 4. AXTARIŞ (R) ----------
docs = []
for c in chunks:
    docs.append(f"[{c['heading']}] {c['text']}")                 # sənədləri embedding-ə çevir
embeddings_docs = embed_model.encode(docs)                    # sənədləri embedding-ə çevir
embeddings_question = embed_model.encode(question)              # SUALI embedding-ə çevir (messages yox!)

scores = util.cos_sim(embeddings_docs,embeddings_question)      
scores = scores.squeeze()   
print(scores)                       # oxşarlıqlar: cos_sim                       # [10, 1] → [10]
top_scores, top_indices = torch.topk(scores, k=5)             # ən yaxşı 3

found_docs = []
found_chunks = []                              # YENİ
for score, idx in zip(top_scores, top_indices):
    if score.item() > THRESHOLD:
        found_docs.append(docs[idx])
        found_chunks.append(chunks[idx])
# ---------- 5. CAVAB (G) ----------
if len(found_docs) == 0:
    print("IDK")
else:
    context = "\n".join(found_docs)
    system_rules = (
        "Answer the question using ONLY the documents provided. "
        "If the answer is not in the documents, say \"I don't know\"."
    )

    user_message = f"Documents:\n{context}\n\nQuestion: {question}"

    messages = [
        {"role": "system", "content": system_rules},   # qaydalar: LLM necə davranmalıdır
        {"role": "user",   "content": user_message},   # data: sənədlər + sual
    ]

    output = llm(messages, max_new_tokens=100)
    answer = output[0]["generated_text"][-1]["content"]

    print("Sual:", question)
    print("Cavab:", answer)
    for c in found_chunks:
        print("Menbeler", c["source"], "|", c["heading"])