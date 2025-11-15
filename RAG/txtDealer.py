'''
专用于处理  名词：解释格式  的txt文件
'''
import re
from bs4 import BeautifulSoup

# 读取原始 txt
with open("glossary_raw.txt", "r", encoding="utf-8") as f:
    text = f.read()

soup = BeautifulSoup(text, "html.parser")

results = []

for p in soup.find_all("p"):
    if not p.find("strong"):
        continue

    strong = p.find("strong")
    term = strong.get_text(" ", strip=True).strip(":")  # 提取术语
    # 删除 <strong>，剩下的文字部分就是定义
    strong.extract()

    desc = p.get_text(" ", strip=True)
    # 如果 strong 之后紧跟一个 ":"，去掉它
    desc = re.sub(r"^[:\s]+", "", desc)

    # 清理 HTML 实体符号
    desc = desc.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")

    results.append((term, desc))

# 输出到文本文件
with open("glossary_clean.txt", "w", encoding="utf-8") as f:
    for term, desc in results:
        f.write(f"{term}: {desc}\n\n")

print(f"✅ 共提取 {len(results)} 个术语，结果已保存到 glossary_clean.txt")
