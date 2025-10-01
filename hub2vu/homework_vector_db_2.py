# -*- coding: utf-8 -*-
import os
import json
import glob
import time
import re
from typing import Iterator, List, Tuple, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI
import chromadb

# ========== 환경 변수 & OpenAI ==========
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
EMBEDDING_MODEL = "text-embedding-3-small"  # 필요시 large로 교체

# ========== Chroma DB ==========
def init_db(db_path: str = "./chroma_db"):
    dbclient = chromadb.PersistentClient(path=db_path)
    try:
        collection = dbclient.get_or_create_collection(name="rag_collection")
    except AttributeError:
        collection = dbclient.create_collection(name="rag_collection", get_or_create=True)
    return dbclient, collection

# ========== 소스 로더 ==========
def iter_txt(folder_path: str) -> Iterator[Tuple[str, str, Dict[str, Any]]]:
    """ .txt 파일을 파일 단위로 스트리밍 """
    for fp in glob.glob(os.path.join(folder_path, "*.txt")):
        filename = os.path.basename(fp)
        with open(fp, "r", encoding="utf-8") as f:
            text = f.read()
        yield (filename, text, {"filename": filename})

def iter_jsonl(folder_path: str) -> Iterator[Tuple[str, str, Dict[str, Any]]]:
    """ .jsonl 파일을 줄 단위로 스트리밍 """
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
                src_id = rec.get("id")
                base_id = f"{filename}::{src_id}" if src_id else filename
                meta = {
                    "filename": filename,
                    "src_id": src_id,
                    "title": rec.get("title"),
                    "section": rec.get("section"),
                    "path": rec.get("path"),
                    "url": rec.get("url"),
                }
                yield (base_id, text, meta)

def iter_sources(folder_path: str) -> Iterator[Tuple[str, str, Dict[str, Any]]]:
    """ txt + jsonl 모두 스트리밍 """
    yield from iter_txt(folder_path)
    yield from iter_jsonl(folder_path)

# ========== 유틸 ==========
def compact_text(s: str, max_chars: int = 200_000) -> str:
    """
    초대용량 텍스트 보호: 공백 축약 + 상한 자르기
    (토큰 상한 여유 있게. 필요시 조정)
    """
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    if len(s) > max_chars:
        s = s[:max_chars]
    return s

def chunk_text_generator(text: str, chunk_size: int = 400, chunk_overlap: int = 50) -> Iterator[str]:
    """ 제너레이터로 청크 생성 (메모리 안전) """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap은 chunk_size보다 작아야 합니다.")
    n = len(text)
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        if chunk.strip():
            yield chunk
        start = end - chunk_overlap
        if start < 0:
            start = 0
        if start >= n:
            break

def embed_batch(texts: List[str], model: str = EMBEDDING_MODEL) -> List[List[float]]:
    """ OpenAI 임베딩 API (단일 배치) """
    for attempt in range(3):
        try:
            resp = client.embeddings.create(model=model, input=texts)
            return [d.embedding for d in resp.data]
        except Exception as e:
            wait = 1.5 * (attempt + 1)
            print(f"[embed retry {attempt+1}/3] {e} -> {wait:.1f}s 대기")
            time.sleep(wait)
    raise RuntimeError("임베딩 API 실패가 계속 발생했습니다.")

# ========== 스트리밍 파이프라인 ==========
def process_sources_streaming(
    folder_path: str,
    collection,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
    BATCH: int = 256
):
    """
    메모리 안전: 청크를 배치 단위로 임베딩 후 곧바로 DB에 add.
    대용량 소스에서도 메모리 사용량이 일정하게 유지됨.
    """
    batch_docs: List[str] = []
    batch_ids: List[str] = []
    batch_meta: List[Dict[str, Any]] = []

    def flush():
        if not batch_docs:
            return
        embs = embed_batch(batch_docs)
        collection.add(
            documents=batch_docs,
            embeddings=embs,
            metadatas=batch_meta,
            ids=batch_ids
        )
        batch_docs.clear()
        batch_ids.clear()
        batch_meta.clear()

    total_chunks = 0
    for base_id, raw_text, meta in iter_sources(folder_path):
        text = compact_text(raw_text)  # 대용량 보호
        for k, chunk in enumerate(chunk_text_generator(text, chunk_size, chunk_overlap)):
            uid = f"{base_id}::chunk{k}"
            m = dict(meta) if meta else {}
            m["chunk_index"] = k

            batch_docs.append(chunk)
            batch_ids.append(uid)
            batch_meta.append(m)
            total_chunks += 1

            if len(batch_docs) >= BATCH:
                flush()

    flush()
    print(f"총 추가된 청크 수: {total_chunks}")

# ========== Entry ==========
def main():
    dbclient, collection = init_db("./chroma_db")
    folder_path = "./source_data"  # 필요시 절대경로로 바꾸세요
    # sanity check
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"폴더가 없습니다: {folder_path}")
    if not (glob.glob(os.path.join(folder_path, "*.txt")) or glob.glob(os.path.join(folder_path, "*.jsonl"))):
        print(f"소스가 없습니다. 폴더를 확인하세요: {folder_path}")
        return

    process_sources_streaming(
        folder_path=folder_path,
        collection=collection,
        chunk_size=400,
        chunk_overlap=50,
        BATCH=256
    )
    print("모든 문서 벡터DB에 저장 완료 ✅")

if __name__ == "__main__":
    main()
