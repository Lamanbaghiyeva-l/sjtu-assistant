RAG over real SJTU documents with sources

What:
Replaced the 10 hard-coded sample sentences with real SJTU documents
loaded from chunks.json (17 chunks from 2 official pages). Each chunk is
embedded as "[heading] text", and every answer now lists its sources
(file + section).

Problems found:
1. Chunk length: 2 chunks (196 and 203 words) exceeded MiniLM's ~256-token
   limit. Fixed by splitting long sections into <=120-word chunks.
2. Retrieval failure: for the internship question, the correct chunk ranked
   4th, so top-3 missed it and the model answered "I don't know".
   Fixed by increasing top-k from 3 to 5.
   Trade-off: only 1 of 5 retrieved chunks contained the answer -> more noise
   for the small LLM and longer prompts.

Tests:
- "How many days before the internship should I apply?" -> 30-45 days (correct)
- "What documents do I need for a stay permit?" -> 4 correct items; top source
  was the stay-permit section, not the residence-permit one (score 0.76)
- "When does the library close?" -> IDK; all scores < 0.3, so the LLM was
  never called

Next: build a 30-question eval set to measure accuracy instead of 3 manual tests.