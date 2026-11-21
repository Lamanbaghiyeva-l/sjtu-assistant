from datetime import date, timedelta
import torch
from transformers import pipeline
import json


def days_before(date_str, days):
    d = date.fromisoformat(date_str)       # string → tarix
    result = d - timedelta(days=days)  # N gün çıx
    
    return result.isoformat()               # tarix → string


def parse_tool_call(answer):
    start = answer.find("{")
    end = answer.rfind("}")

    if start == -1:                  # "{" tapılmadı
        return None

    json_text = answer[start : end + 1]
    data = json.loads(json_text)

    if "tool" in data:
        return data
    return None

system_rules = """You are a helpful assistant. You have one tool:

days_before(date_str, days): returns the date that is `days` days before `date_str`.
  date_str format: YYYY-MM-DD

If you need the tool, reply with ONLY this JSON and nothing else:
{"tool": "days_before", "date_str": "YYYY-MM-DD", "days": N}

If you don't need the tool, answer normally."""

llm = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype="auto",
    device=0,
)

questions = [
    "My residence permit expires on 2026-12-15. I must apply at least 30 days before. What is the latest date to apply?",
    "What is the capital of France?",
]

for q in questions:
    messages = [
        {"role": "system", "content": system_rules},
        {"role": "user",   "content": q}
    ]
    output = llm(messages, max_new_tokens=100)
    answer = output[0]["generated_text"][-1]["content"]
    call = parse_tool_call(answer)          # ← dövrün içində
    print("Sual:", q)
    if call is None:
        print("Cavab:", answer)
    else:
        result = days_before(call["date_str"], call["days"])
        print("Tool nəticəsi:", result)

         # ↓↓↓ YENİ ↓↓↓
        messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user", "content": f"Tool result: {result}. Now answer the original question in one sentence."})

        output2 = llm(messages, max_new_tokens=100)
        final = output2[0]["generated_text"][-1]["content"]
        print("Son cavab:", final)
        # ↑↑↑ YENİ ↑↑↑
    print()
   



