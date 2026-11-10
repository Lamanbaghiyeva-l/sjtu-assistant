import torch
from sentence_transformers import SentenceTransformer, util

# ---------- 1. MODEL ----------
model_path = "/mnt/pool1/sharehome/laman/workspace/resume_screening/models/all-MiniLM-L6-v2"
model = SentenceTransformer(model_path)

# ---------- 2. SƏNƏDLƏR (sınaq üçün nümunə mətnlər) ----------
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
# question = "What is the capital of France?"     # IDK sınağı üçün bu sətri aç

# ---------- 4. EMBEDDING ----------
embeddings_docs = model.encode(docs)              # [10, 384]
embeddings_question = model.encode(question)      # [384]

# ---------- 5. AXTARIŞ ----------
THRESHOLD = 0.3

scores = util.cos_sim(embeddings_docs, embeddings_question)   # [10, 1]
scores = scores.squeeze()                                     # [10]

top_scores, top_indices = torch.topk(scores, k=3)             # ən yüksək 3 bal və yerləri

# ---------- 6. CAVAB ----------
print("Sual:", question)

if top_scores[0].item() < THRESHOLD:             # ən yüksək bal belə aşağıdırsa
    print("IDK")
else:
    for score, idx in zip(top_scores, top_indices):
        if score.item() > THRESHOLD:
            print(round(score.item(), 2), docs[idx])