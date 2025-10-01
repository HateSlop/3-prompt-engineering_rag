# -*- coding: utf-8 -*-
import os
import json
import glob
import time
from typing import List, Tuple, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI
import chromadb

# =========================
# 1) 환경 변수 & OpenAI 클라이언트
# =========================

# .env 로드 (현재 폴더 우선, 없으면 상위 폴더 시도)
if os.path.exists("./.env"):
    load_dotenv("./.env")
elif os.path.exists("../.env"):
    load_dotenv("../.env")
else:
    load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("API Key가 설정되지 않았습니다. .env의 OPENAI_API_KEY를 확인하세요.")

client = OpenAI(api_key=API_KEY)

# 임베딩 모델 (비용/성능에 따라 선택)
EMBEDDING_MODEL = "text-embedding-3-small"  # 또는 "text-embedding-3-large"

# =========================
# 2) Chroma DB 초기화
# =========================
def init_db(db_path: str = "./chroma_db"):
    """
    로컬 경로에 Chroma Persistent DB 초기화.
    """
    dbclient = chromadb.PersistentClient(path=db_path)
    # 최신 chromadb는 get_or_create_collection 사용 권장
    try:
        collection = dbclient.get_or_create_collection(name="rag_collection")
    except AttributeError:
        # 구버전 호환
        collection = dbclient.create_collection(name="rag_collection", get_or_create=True)
    return dbclient, collection

# =========================
# 3) 소스 로더
# =========================
def load_text_files(folder_path: str) -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    .txt 파일을 (base_id, text, metadata) 리스트로 반환.
    base_id: filename
    metadata: {"filename": filename}
    """
    items = []
    for fp in glob.glob(os.path.join(folder_path, "*.txt")):
        filename = os.path.basename(fp)
        with open(fp, "r", encoding="utf-8") as f:
            text = f.read()
        items.append((filename, text, {"filename": filename}))
    return items

def load_jsonl_files(folder_path: str) -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    .jsonl 파일을 (base_id, text, metadata) 리스트로 반환.
    각 라인은 최소 {"text": "..."} 필요.
    base_id: filename::rec_id(있으면)
    metadata: {"filename", "src_id"(jsonl의 id), "title","section","path","url" 등}
    """
    items = []
    for fp in glob.glob(os.path.join(folder_path, "*.jsonl")):
        filename = os.path.basename(fp)
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = rec.get("text", "")
                if not text.strip():
                    continue
                src_id = rec.get("id")  # 섹션/레코드 ID라면 활용
                base_id = f"{filename}::{src_id}" if src_id else filename
                meta = {
                    "filename": filename,
                    "src_id": src_id,
                    "title": rec.get("title"),
                    "section": rec.get("section"),
                    "path": rec.get("path"),
                    "url": rec.get("url"),
                }
                items.append((base_id, text, meta))
    return items

def load_sources(folder_path: str) -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    폴더에서 .txt + .jsonl 모두 로드하여 하나의 리스트로 반환.
    """
    items = []
    items.extend(load_text_files(folder_path))
    items.extend(load_jsonl_files(folder_path))
    return items

# =========================
# 4) 청크 유틸
# =========================
def chunk_text(text: str, chunk_size: int = 400, chunk_overlap: int = 50) -> List[str]:
    """
    단순 문자 길이 기준 청크. (필요시 문장/토큰 기준으로 개선 가능)
    """
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        # 다음 시작점(오버랩 적용)
        start = end - chunk_overlap
        if start < 0:
            start = 0
        if start >= n:
            break
    return chunks

# =========================
# 5) 임베딩 API (배치)
# =========================
def embed_texts(texts: List[str], model: str = EMBEDDING_MODEL, batch_size: int = 64) -> List[List[float]]:
    """
    OpenAI 임베딩 API를 배치로 호출하여 리스트 반환.
    - texts: 임베딩할 문자열 리스트
    - model: 임베딩 모델명
    - batch_size: 한 번에 보낼 샘플 수
    """
    vectors: List[List[float]] = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        # API 호출 (재시도 간단 처리)
        for attempt in range(3):
            try:
                resp = client.embeddings.create(model=model, input=batch)
                vectors.extend([d.embedding for d in resp.data])
                break
            except Exception as e:
                # 간단 백오프
                wait = 1.5 * (attempt + 1)
                print(f"[embed retry {attempt+1}/3] {e} -> {wait:.1f}s 대기")
                time.sleep(wait)
        else:
            # 3회 실패 시 빈 벡터로 채우고 지속 (또는 raise)
            raise RuntimeError("임베딩 API 실패가 계속 발생했습니다.")
    return vectors

# =========================
# 6) 메인 파이프라인
# =========================
def main():
    dbclient, collection = init_db("./chroma_db")
    folder_path = "./source_data"
    sources = load_sources(folder_path)
    if not sources:
        print(f"소스가 없습니다. 폴더를 확인하세요: {folder_path}")
        return

    # 청크 생성 및 ID/메타 생성
    doc_records: List[str] = []        # documents
    meta_records: List[Dict[str, Any]] = []
    id_records: List[str] = []

    # 전역 고유 ID 방지: base_id + chunk_index
    for base_id, text, meta in sources:
        chunks = chunk_text(text, chunk_size=400, chunk_overlap=50)
        for k, chunk in enumerate(chunks):
            uid = f"{base_id}::chunk{k}"
            doc_records.append(chunk)
            m = dict(meta) if meta else {}
            m["chunk_index"] = k
            meta_records.append(m)
            id_records.append(uid)

    if not doc_records:
        print("청크로 나눌 텍스트가 없습니다.")
        return

    print(f"총 문서 청크 수: {len(doc_records)}")

    # 임베딩 API 호출 (배치)
    embeddings = embed_texts(doc_records, model=EMBEDDING_MODEL, batch_size=64)

    # Chroma에 배치 추가 (한 번에 너무 많이 넣는 경우 잘라서)
    BATCH = 500
    for i in range(0, len(doc_records), BATCH):
        j = i + BATCH
        collection.add(
            documents=doc_records[i:j],
            embeddings=embeddings[i:j],
            metadatas=meta_records[i:j],
            ids=id_records[i:j],
        )
        print(f"추가 완료: {i} ~ {j-1}")

    print("모든 문서 벡터DB에 저장 완료 ✅")

# =========================
# Entry
# =========================
if __name__ == "__main__":
    main()
