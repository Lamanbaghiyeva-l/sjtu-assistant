from bs4 import BeautifulSoup

def clean_html(input_path, class_name, output_path):
    html = open(input_path, encoding="utf-8").read()
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find("div", class_= class_name)

    blocks = article.find_all(["p", "li", "h2"])
    lines = []
    for b in blocks:
        line = b.get_text(" ", strip = True)

        if line:
            lines.append(" ".join(line.split()))
    
    clean_text = "\n".join(lines)

    open(output_path, "w", encoding="utf-8").write(clean_text)
    print("Saxlandı:", output_path, "-", len(lines), "sətir")


clean_html("data/internship.html", "article_content", "data/internship.txt")
clean_html("data/residence_permit.html", "Article_content", "data/residence_permit.txt")