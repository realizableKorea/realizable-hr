import os
import pandas as pd
import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup

# ✅ ChromaDB 설정
DB_PATH = "./chroma_db_v3"
COLLECTION_NAME = "company_docs"
DOCS_PATH = "./realizable_markdown"

# ✅ ChromaDB 인스턴스 생성
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# ✅ 임베딩 모델 로드
model_1 = 'all-MiniLM-L6-v2'
model_2 = 'upstage/solar-embedding-1-large'
model_3 = 'bge-large-en' # 최근 인기, 검색에 특화된 성능 우수 모델, 다소 무거움
model_4 = 'all-mpnet-base-v2' 
embedding_model = SentenceTransformer(model_4)

def extract_text_from_html(file_path):
    """ HTML 파일에서 텍스트 추출 """
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
    return soup.get_text(separator=" ", strip=True)

def extract_text_from_csv(file_path):
    """ CSV 파일에서 텍스트 추출 """
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="latin-1")

    text_data = df.astype(str).apply(lambda x: ' '.join(x.dropna()), axis=1).tolist()
    return " ".join(text_data)  # CSV의 모든 텍스트를 하나의 문자열로 합침

def extract_text_from_pdf(file_path):
    """ PDF 파일에서 텍스트 추출 """
    text = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text.append(page.extract_text() or "")
    return " ".join(text).strip()

def extract_text_from_md(file_path):
    """ Markdown (.md) 파일에서 텍스트 추출 """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text.strip()  # 불필요한 공백 제거

def get_all_files(directory, extensions):
    """ 특정 확장자의 파일을 재귀적으로 검색 """
    matched_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(extensions):
                matched_files.append(os.path.join(root, file))
    return matched_files

def embed_documents():
    """ HTML, CSV, PDF, Markdown 문서를 벡터로 변환하고 ChromaDB에 저장 """
    docs_path = DOCS_PATH
    # html_files = get_all_files(docs_path, (".html", ".htm"))
    # csv_files = get_all_files(docs_path, (".csv",))
    # pdf_files = get_all_files(docs_path, (".pdf",))
    md_files = get_all_files(docs_path, (".md",))  # ✅ Markdown 파일 추가

    # all_files = html_files + csv_files + pdf_files + md_files
    # if not all_files:
    #     print("❌ No HTML, CSV, PDF, or Markdown files found in docs directory!")
    #     return

    print(f"🔍 Found {len(md_files)} files (HTML, CSV, PDF, Markdown). Processing...")

    for file_path in md_files:
        doc_id = os.path.basename(file_path)  # 파일명을 ID로 사용

        # 파일 유형별로 처리
        if file_path.endswith((".html", ".htm")):
            text = extract_text_from_html(file_path)
        elif file_path.endswith(".csv"):
            text = extract_text_from_csv(file_path)
        elif file_path.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        elif file_path.endswith(".md"):  # ✅ Markdown 파일 처리
            text = extract_text_from_md(file_path)
        else:
            continue

        if text and isinstance(text, str) and len(text.strip()) > 20 and not pd.isna(text) and not isinstance(text, float):
            embedding = embedding_model.encode(text).tolist()
            collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[text],  # ✅ 원본 텍스트 저장
                metadatas=[{"source": file_path}]
            )
            print(f"✅ Embedded {file_path}")
        else:
            print(f"⚠️ Skipping {file_path} (text too short or None)")


# ✅ 실행
# collection.delete()  # 모든 데이터 삭제
# print("🚀 ChromaDB 모든 데이터 삭제 완료!")

# ✅ 데이터 재삽입 실행
embed_documents()
print("✅ 데이터 재학습 완료!")
print("collection count: " + str(collection.count()))
# print("sample_documents below")
# print(str(collection.peek(limit=1)))
