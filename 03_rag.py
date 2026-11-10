import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# ---------- 1. MODELLƏR ----------
embed_model = SentenceTransformer("/mnt/pool1/sharehome/laman/workspace/resume_screening/models/all-MiniLM-L6-v2")

llm = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype="auto",
    device=0,
)

# ---------- 2. SƏNƏDLƏR ----------
docs = [
    "The library is open from 8 am to 10 pm.",
    "International students must register their residence permit within 30 days.",
    "The cafeteria serves halal food on the second floor.",
    "Students can borrow up to 10 books from the library for 30 days.",
    "The gym is free for students and opens at 6 am.",
    "To get an internship permit, international students apply through the International Student Office.",
    "Campus buses run every 15 minutes between the dormitories and the main buildings.",
    "Visa extensions must be requested at least one month before the visa expires.",
    "The library has quiet study rooms that can be booked online.",
    "Halal and vegetarian meals are also available at the east canteen.",
]

# ---------- 3. SUAL ----------
question = "When does the library close?"
THRESHOLD = 0.3

# ---------- 4. AXTARIŞ (R) ----------
embeddings_docs = embed_model.encode(docs)                    # sənədləri embedding-ə çevir
embeddings_question = embed_model.encode(question)              # SUALI embedding-ə çevir (messages yox!)

scores = util.cos_sim(embeddings_docs,embeddings_question)      
scores = scores.squeeze()   
print(scores)                       # oxşarlıqlar: cos_sim                       # [10, 1] → [10]
top_scores, top_indices = torch.topk(scores, k=3)             # ən yaxşı 3

found_docs = []                          # boş səbət
for score, idx in zip(top_scores, top_indices):
    if score.item() > THRESHOLD:                  # bal həddən yuxarıdırsa
        found_docs.append(docs[idx])        # səbətə əlavə et

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
    print("Tapılan sənədlər:", found_docs)
    print("Cavab:", answer)