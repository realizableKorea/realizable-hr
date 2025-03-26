from fastapi import FastAPI, Query
from fastapi import APIRouter
import chromadb
import os
from sentence_transformers import SentenceTransformer
import pandas as pd
import pdfplumber
from bs4 import BeautifulSoup


router = APIRouter()
# DB_PATH = "./chroma_db_v3"
DB_PATH = "./chroma_db_v3"
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(name="company_docs")

# ✅ 임베딩 모델 로드
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
model_4 = 'all-mpnet-base-v2' 
embedding_model = SentenceTransformer(model_4)

def query_embedding(query: str):
    """ 사용자의 질문을 벡터로 변환 """
    return embedding_model.encode(query).tolist()

def search_documents(query: str, top_n: int = 10, min_score: float = 0.3):
    """ ChromaDB에서 가장 유사한 문서를 검색 """
    query_vector = query_embedding(query)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_n,
        # include_distances=True
    )
    return results

def generate_response(query: str, results):
    """ 검색된 문서를 기반으로 응답 생성 """
    if not results or "documents" not in results or not results["documents"]:
        return "죄송합니다. 해당 질문에 대한 정보를 찾을 수 없습니다."

    # ✅ 중첩 리스트에서 `None` 값 제거 (documents는 `list of lists` 형태)
    relevant_docs = [
        doc for doc_list in results["documents"] for doc in doc_list if doc is not None
    ]

    if not relevant_docs:  # 모든 값이 None이면 예외 처리
        return "죄송합니다. 관련된 정보를 찾을 수 없습니다."

    context = "\n".join(relevant_docs)

    response = f"📌 [HR Bot] 다음은 '{query}'에 대한 정보입니다:\n{context}"
    return response


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

def get_all_files(directory, extensions):
    """ 특정 확장자의 파일을 재귀적으로 검색 """
    matched_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(extensions):
                matched_files.append(os.path.join(root, file))
    return matched_files

def embed_documents():
    """ HTML, CSV, PDF 문서를 벡터로 변환하고 ChromaDB에 저장 """
    docs_path = "./realizable_internal_data"
    html_files = get_all_files(docs_path, (".html", ".htm"))
    csv_files = get_all_files(docs_path, (".csv",))
    pdf_files = get_all_files(docs_path, (".pdf",))

    all_files = html_files + csv_files + pdf_files
    if not all_files:
        print("❌ No HTML, CSV, or PDF files found in docs directory!")
        return

    print(f"🔍 Found {len(all_files)} files (HTML, CSV, PDF). Processing...")

    for file_path in all_files:
        doc_id = os.path.basename(file_path)  # 파일명을 ID로 사용

        # 파일 유형별로 처리
        if file_path.endswith((".html", ".htm")):
            text = extract_text_from_html(file_path)
        elif file_path.endswith(".csv"):
            text = extract_text_from_csv(file_path)
        elif file_path.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        else:
            continue

        if len(text) > 20:  # 너무 짧은 문서는 제외
            embedding = embedding_model.encode(text).tolist()
            collection.add(ids=[doc_id], embeddings=[embedding], metadatas=[{"source": file_path}])
            print(f"✅ Embedded {file_path}")
        else:
            print(f"⚠️ Skipping {file_path} (too short)")

@router.get("/ask/")
def ask_hr_bot(question: str = Query(..., title="질문", description="HR 관련 질문을 입력하세요")):
    """ 질문에 기반한 기업 내부 정보를 답변 제공 """
    results = search_documents(question)
    answer = generate_response(question, results)
    return {"question": question, "answer": answer}

@router.get("/count_study_data")
def collection_count():
    """ 현재 저장된 문서 개수 응답 """
    return {"현재 저장된 문서 개수": collection.count()}

@router.get("/show_sample_data")
def get_sample():
    """ 학습한 데이터중 샘플 3개를 응답함 """
    return {"sample_documents": str(collection.peek(limit=10))}

@router.get("/store_embedding")
def study_start():
    """ embedding을 시작함 """
    embed_documents()
    print("collection count: " + str(collection.count()))
    # print("sample_documents below")
    # print(collection.peek(limit=1))
    print("🚀 All documents (HTML, CSV, PDF) embedded into ChromaDB successfully!")
    return {"msg":"🚀 All documents (HTML, CSV, PDF) embedded into ChromaDB successfully!",\
            "embedded data count: ":collection.count()}


def search_docs(query, top_k=5):
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    # ✅ 중첩 리스트를 평탄화하여 1차원 리스트로 변환
    retrieved_docs = sum(results["documents"], [])

    return retrieved_docs

import ollama

def generate_answer_with_rag(query, retrieved_docs):
    # ✅ 리스트를 문자열로 변환하여 문제 해결
    context = "\n".join(retrieved_docs)
    
    prompt = f"다음 문서를 기반으로 질문에 답하세요:\n\n{context}\n\n질문: {query}\n답변: "
    
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    return response['message']['content']

@router.get("/rag")
def rag_api(query: str = Query(..., description="사용자 질문")):
    retrieved_docs = search_docs(query)
    response = generate_answer_with_rag(query, retrieved_docs)
    
    return {"answer": response}