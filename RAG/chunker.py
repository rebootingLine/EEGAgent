import fitz
import re
from nltk.tokenize import sent_tokenize

def extract_clean_text(filepath, header_lines_num=2, footer_lines_num=2):
    """
    读取PDF，去除每页前后页眉页脚，合并成全文，
    并截断Reference之后的内容。
    """
    doc = fitz.open(filepath)
    all_text_lines = []

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]
        lines = []
        for block in blocks:
            if block["type"] != 0:
                continue
            lines.extend(block["lines"])

        page_lines_text = []
        for line in lines:
            text = "".join([s["text"] for s in line["spans"]]).strip()
            if text:
                page_lines_text.append(text)

        if len(page_lines_text) > header_lines_num + footer_lines_num:
            page_lines_text = page_lines_text[header_lines_num : -footer_lines_num]
        elif len(page_lines_text) > header_lines_num:
            page_lines_text = page_lines_text[header_lines_num:]

        all_text_lines.extend(page_lines_text)

    full_text = "\n".join(all_text_lines)

    ref_match = re.search(r"\n(references|reference|bibliography)\s*\n", full_text, re.IGNORECASE)
    if ref_match:
        full_text = full_text[:ref_match.start()]

    return full_text

def chunk_text_by_sentences(text, max_sentences=5, max_chars=500):
    """
    按句子拆分文本，并将句子合并为长度适中的块
    """
    text = re.sub(r'\n+', ' ', text).strip() # 避免换行被认为是新起一句.
    sentences = sent_tokenize(text, language='english')
    chunks = []
    current_chunk = []
    current_len = 0

    for sent in sentences:
        if len(current_chunk) >= max_sentences or (current_len + len(sent)) > max_chars:
            if current_chunk:  # 确保当前块不为空
                chunks.append(" ".join(current_chunk))
            current_chunk = [sent]
            current_len = len(sent)
        else:
            current_chunk.append(sent)
            current_len += len(sent)

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def load_and_chunk(filepath, max_length=500):
    """
    加载 PDF 或 TXT 文件，并按段落或句子切分为 chunks。
    PDF -> 用 extract_clean_text()
    TXT -> 按行（或空行）切分
    """
    def chunk_by_lines(text):
        # 去掉多余空行，并按段落切分
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        return paragraphs

    if filepath.endswith('.pdf'):
        text = extract_clean_text(filepath)
        return chunk_text_by_sentences(text)

    elif filepath.endswith('.txt'):
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        chunks = chunk_by_lines(text)
        return chunks

    else:
        raise ValueError("Only PDF and TXT files are supported.")
