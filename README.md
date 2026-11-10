# SJTU Student Assistant

A question-answering assistant for SJTU students, built step by step with Retrieval-Augmented Generation (RAG).
The goal: a student asks a question ("When does the library close?") and the assistant answers using university documents instead of guessing.

> Status: early prototype. The documents are 10 short sample sentences written for testing, not official SJTU information yet.

## How it works

```
question ──► MiniLM embedding ──► find the 3 most similar documents ──► Qwen LLM ──► answer
                                   (cosine similarity + threshold)       (uses only these documents)
```

1. **Retrieve (R):** every document and the question are turned into 384-number embeddings with `all-MiniLM-L6-v2`. The top 3 documents by cosine similarity are kept if their score is above a threshold (0.3).
2. **Generate (G):** the kept documents are given to `Qwen2.5-1.5B-Instruct` with a rule to answer only from them. If no document passes the threshold, the assistant says "I don't know".

## Files

| File | What it does |
|---|---|
| `01_semantic_search.py` | Embeddings, cosine similarity, top-3 search and the "I don't know" threshold |
| `02_llm_hello.py` | Loads Qwen2.5-1.5B-Instruct with a Hugging Face `pipeline` and asks one question |
| `03_rag.py` | Full RAG: search + LLM answer |

## Example

```
Question: When does the library close?
Found:    The library is open from 8 am to 10 pm. (+ 2 other library documents)
Answer:   The library closes at 10 pm.
```

## What I learned

- **Meaning beats exact words.** "Python developer" and "Python programmer" scored 0.89 similarity, while a keyword match would treat them as different.
- **There is always a "best" document**, even for unrelated questions ("What is the capital of France?" still had a top match at 0.05). A threshold lets the system say "I don't know" instead of answering from the wrong document.
- **Retrieval finds related documents; the LLM picks the answer.** For the library question all 3 results were about the library, but only one answered it, and the LLM used that one.

## How to run

```bash
pip install torch sentence-transformers transformers
python 03_rag.py
```

The MiniLM model path in the scripts points to a local folder; change it to `"sentence-transformers/all-MiniLM-L6-v2"` to download it from Hugging Face. Tested on an NVIDIA RTX 6000 Ada (the LLM runs on GPU 0).

## Next steps

- [ ] Real SJTU documents (international student rules, library, dormitories)
- [ ] Chunking long documents into small pieces
- [ ] Vector database (FAISS or Chroma) to store embeddings once
- [ ] Evaluation with a set of test questions
- [ ] FastAPI endpoint `POST /ask`
- [ ] Docker
