# Enhanced RAG

프롬프트 엔지니어링을 적용한 RAG 시스템

## 파일

- `enhanced_rag.py` - 개선된 RAG (Query Expansion + Re-ranking + CoT)
- `rag_chatbot.py` - 기존 RAG (비교용)
- `build_vector_db.py` - DB 구축 (이미 완료)

## 개선 사항

1. **Query Expansion**: 질문을 여러 방식으로 재표현하여 검색
2. **Re-ranking**: GPT로 문서 관련성 재평가
3. **Chain-of-Thought**: 단계적 추론 (핵심답변 -> 근거 -> 추가정보 -> 한계)

## 실행

```bash
# 기존 chroma_db 사용 (이미 구축됨)
python enhanced_rag.py
```

## 비교

| 기존 | 개선 |
|------|------|
| 단일 쿼리 검색 | Query Expansion |
| 유사도만 사용 | Re-ranking 추가 |
| 기본 프롬프트 | CoT 프롬프트 |
