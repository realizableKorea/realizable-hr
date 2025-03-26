import os
import chromadb
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM
import markdown

# 🔹 설정
DOCS_PATH = "./realizable_markdown"
DB_PATH = "./chroma_db_gemma"
COLLECTION_NAME = "company_docs"
# EMBEDDING_MODEL_NAME = "upstage/SOLAR-embedding-1-large"  # or "all-mpnet-base-v2"
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"  # or "all-mpnet-base-v2"
LLM_MODEL_NAME = "google/gemma-2b"

# 🔹 ChromaDB 초기화
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# 🔹 임베딩 모델 불러오기
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# 🔹 문서 전처리 함수
def extract_text_from_markdown(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        html = markdown.markdown(f.read())
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text()

# 🔹 문서 임베딩 및 ChromaDB에 추가
def embed_all_documents():
    for filename in os.listdir(DOCS_PATH):
        if filename.endswith(".md"):
            file_path = os.path.join(DOCS_PATH, filename)
            text = extract_text_from_markdown(file_path)
            embedding = embedding_model.encode([text])[0]  # 단일 문서
            collection.add(
                documents=[text],
                embeddings=[embedding.tolist()],
                ids=[filename]
            )
    print("✅ 모든 Markdown 문서 임베딩 완료")

# 🔹 LLM 준비 (Gemma)
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, device_map="auto")
    return tokenizer, model

# 🔹 질문에 답변 생성
def answer_question(user_query, top_k=3):
    # 1. 유사 문서 검색
    query_embedding = embedding_model.encode([user_query])[0]
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

    # 2. 관련 문서 내용 합치기
    context = "\n\n".join(results['documents'][0])

    # 3. LLM에 질의 생성
    tokenizer, model = load_llm()
    prompt = f"""당신은 문서를 기반으로 질문에 답하는 AI입니다.
아래는 문서 내용입니다:

{context}

질문: {user_query}
답변:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=300)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("📝 답변:\n", answer)

# 🔹 실행 예시
if __name__ == "__main__":
    embed_all_documents()
    question = input("질문을 입력하세요: ")
    answer_question(question)
