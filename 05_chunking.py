import json


def add_chunks(chunks, lines, heading, path):
    group = []     # kiçik qutu
    count = 0      # qutudakı söz sayı

    for line in lines:
        # 1. bu sətrin söz sayı
        len_line = len(line.split())

        # 2. qutu dolacaqsa → əvvəlcə köhnə qutunu rəfə qoy
        if count + len_line > 120 and group:
            chunks.append({"source": path, "heading": heading, "text": " ".join(group)})
            group = []
            count = 0

        # 3. sətri qutuya at və sayı artır
        group.append(line)
        count += len_line

    # 4. son qutu
    if group:
        chunks.append({"source": path, "heading": heading, "text": " ".join(group)})


def chunk_file(path):
    text = open(path, encoding="utf-8").read()
    lines = text.split("\n")

    SMALL_HEADINGS = ["Required materials", "Application process", "Notes"]

    chunks = []
    current_section = "Intro"
    current_heading = "Intro"
    current_lines = []

    for line in lines:
        if len(line.split()) <= 8 and not line.endswith(".") and not line.startswith("http"):
            # 1. köhnə qutunu rəfə qoy (lazım olsa, hissələrə bölərək)
            if current_lines:
                add_chunks(chunks, current_lines, current_heading, path)

            # 2. yeni etiket qur
            if line in SMALL_HEADINGS:
                current_heading = current_section + " > " + line
            else:
                current_section = line
                current_heading = line

            # 3. qutunu boşalt
            current_lines = []
        else:
            current_lines.append(line)

    # son qutu
    add_chunks(chunks, current_lines, current_heading, path)

    return chunks


all_chunks = chunk_file("data/residence_permit.txt") + chunk_file("data/internship.txt")

for c in all_chunks:
    print(len(c["text"].split()), "söz |", c["heading"])
print("Cəmi:", len(all_chunks), "chunk")

json.dump(all_chunks, open("data/chunks.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("Saxlandı: data/chunks.json")
