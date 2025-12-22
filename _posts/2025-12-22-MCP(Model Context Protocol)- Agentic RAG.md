---
layout: post
title: MCP(Model Context Protocol) - Agentic RAG
summary: MCP(Model Context Protocol) - Agentic RAG
author: keonhee
date: 2025-12-22 10:00:00 +0900
category: MCP
keywords: MCP
permalink: /blog/MCP_6/
usemathjax: true
thumbnail: /assets/img/posts/overview_of_NLP_1.png
---
# 06. 실전 프로젝트: RAG를 MCP로 만들기

## 목표
여러분이 LangGraph로 만들었던 RAG 시스템을 MCP 서버로 전환해봅니다.

---

## 프로젝트 개요

### 만들 것: 문서 검색 RAG 서버

```
📚 Document RAG Server
├── upload_document    - 문서 업로드 및 벡터화
├── search_documents   - 유사 문서 검색
├── list_documents     - 전체 문서 목록
└── delete_document    - 문서 삭제
```

### 기술 스택
- **임베딩**: Sentence Transformers (로컬 실행)
- **벡터 저장**: JSON (간단한 구현)
- **검색**: 코사인 유사도

---

## 단계 1: 필요한 패키지 설치

```bash
pip install sentence-transformers numpy
```

**각 패키지의 역할:**
- `sentence-transformers`: 문장을 벡터로 변환
- `numpy`: 벡터 연산 (유사도 계산)

---

## 단계 2: RAG 서버 뼈대 만들기

`servers/rag_server.py` 파일 생성:

```python
"""
문서 검색 RAG MCP 서버
문서를 저장하고 검색하는 기능을 제공합니다.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio

# 서버 인스턴스
app = Server("rag-server")

# 임베딩 모델 로드 (처음엔 시간이 걸릴 수 있어요)
print("임베딩 모델 로딩 중...")
embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
print("모델 로딩 완료!")

# 문서 저장 경로
DOCS_DIR = Path("rag_documents")
DOCS_DIR.mkdir(exist_ok=True)

INDEX_FILE = DOCS_DIR / "index.json"
```

---

## 단계 3: 헬퍼 함수들 작성

같은 파일에 계속:

```python
def load_index() -> Dict[str, Any]:
    """인덱스 파일 로드"""
    if not INDEX_FILE.exists():
        return {"documents": []}
    
    with open(INDEX_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_index(index: Dict[str, Any]):
    """인덱스 파일 저장"""
    with open(INDEX_FILE, 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def add_document_to_index(doc_id: str, title: str, content: str, embedding: List[float]):
    """문서를 인덱스에 추가"""
    index = load_index()
    
    document = {
        "id": doc_id,
        "title": title,
        "content": content,
        "embedding": embedding,
        "created_at": datetime.now().isoformat()
    }
    
    index["documents"].append(document)
    save_index(index)
    
    return document


def search_similar_documents(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """쿼리와 유사한 문서 검색"""
    index = load_index()
    
    if not index["documents"]:
        return []
    
    # 쿼리를 벡터로 변환
    query_embedding = embedder.encode(query).tolist()
    
    # 각 문서와의 유사도 계산
    results = []
    for doc in index["documents"]:
        doc_embedding = doc["embedding"]
        
        # 코사인 유사도 계산
        similarity = cosine_similarity(query_embedding, doc_embedding)
        
        results.append({
            "document": doc,
            "similarity": similarity
        })
    
    # 유사도 순으로 정렬
    results.sort(key=lambda x: x["similarity"], reverse=True)
    
    # 상위 k개만 반환
    return results[:top_k]


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """코사인 유사도 계산"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    return dot_product / (norm1 * norm2)


def get_all_documents() -> List[Dict[str, Any]]:
    """모든 문서 목록"""
    index = load_index()
    
    # 임베딩 제외하고 반환 (너무 길어서)
    return [
        {
            "id": doc["id"],
            "title": doc["title"],
            "created_at": doc["created_at"],
            "content_preview": doc["content"][:100] + "..."
        }
        for doc in index["documents"]
    ]


def delete_document_by_id(doc_id: str) -> bool:
    """문서 삭제"""
    index = load_index()
    
    original_count = len(index["documents"])
    index["documents"] = [
        doc for doc in index["documents"]
        if doc["id"] != doc_id
    ]
    
    if len(index["documents"]) < original_count:
        save_index(index)
        return True
    
    return False
```

---

## 단계 4: MCP 도구 정의

```python
@app.list_tools()
async def list_tools() -> list[Tool]:
    """제공하는 도구 목록"""
    return [
        # 1. 문서 업로드
        Tool(
            name="upload_document",
            description="새로운 문서를 업로드하고 벡터 인덱스에 추가합니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "문서의 고유 ID"
                    },
                    "title": {
                        "type": "string",
                        "description": "문서 제목"
                    },
                    "content": {
                        "type": "string",
                        "description": "문서 내용"
                    }
                },
                "required": ["doc_id", "title", "content"]
            }
        ),
        
        # 2. 문서 검색
        Tool(
            name="search_documents",
            description="쿼리와 의미적으로 유사한 문서를 검색합니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "검색 쿼리"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "반환할 문서 개수 (기본값: 3)",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        ),
        
        # 3. 문서 목록
        Tool(
            name="list_documents",
            description="저장된 모든 문서의 목록을 반환합니다",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        
        # 4. 문서 삭제
        Tool(
            name="delete_document",
            description="특정 문서를 삭제합니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "삭제할 문서의 ID"
                    }
                },
                "required": ["doc_id"]
            }
        )
    ]
```

---

## 단계 5: 도구 실행 로직

```python
@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """도구 실행"""
    
    if name == "upload_document":
        # 문서 업로드 및 벡터화
        doc_id = arguments["doc_id"]
        title = arguments["title"]
        content = arguments["content"]
        
        # 문서를 벡터로 변환
        embedding = embedder.encode(content).tolist()
        
        # 인덱스에 추가
        doc = add_document_to_index(doc_id, title, content, embedding)
        
        return [
            TextContent(
                type="text",
                text=f"✅ 문서 업로드 완료!\n"
                     f"ID: {doc['id']}\n"
                     f"제목: {doc['title']}\n"
                     f"내용 길이: {len(content)} 글자\n"
                     f"임베딩 차원: {len(embedding)}"
            )
        ]
    
    elif name == "search_documents":
        # 문서 검색
        query = arguments["query"]
        top_k = arguments.get("top_k", 3)
        
        results = search_similar_documents(query, top_k)
        
        if not results:
            return [
                TextContent(
                    type="text",
                    text="🔍 검색 결과가 없습니다. 먼저 문서를 업로드해주세요."
                )
            ]
        
        # 결과 포맷팅
        result_text = f"🔍 '{query}' 검색 결과 (상위 {len(results)}개):\n\n"
        
        for i, result in enumerate(results, 1):
            doc = result["document"]
            similarity = result["similarity"]
            
            result_text += f"{i}. 📄 {doc['title']}\n"
            result_text += f"   유사도: {similarity:.4f}\n"
            result_text += f"   내용: {doc['content'][:200]}...\n"
            result_text += f"   ID: {doc['id']}\n\n"
        
        return [
            TextContent(
                type="text",
                text=result_text
            )
        ]
    
    elif name == "list_documents":
        # 문서 목록
        documents = get_all_documents()
        
        if not documents:
            return [
                TextContent(
                    type="text",
                    text="📭 저장된 문서가 없습니다."
                )
            ]
        
        doc_list = "\n".join([
            f"- {doc['id']}: {doc['title']} (생성: {doc['created_at']})"
            for doc in documents
        ])
        
        return [
            TextContent(
                type="text",
                text=f"📚 문서 목록 ({len(documents)}개):\n{doc_list}"
            )
        ]
    
    elif name == "delete_document":
        # 문서 삭제
        doc_id = arguments["doc_id"]
        
        if delete_document_by_id(doc_id):
            return [
                TextContent(
                    type="text",
                    text=f"🗑️ 문서 삭제 완료: {doc_id}"
                )
            ]
        else:
            return [
                TextContent(
                    type="text",
                    text=f"❌ 문서를 찾을 수 없습니다: {doc_id}"
                )
            ]
    
    else:
        raise ValueError(f"알 수 없는 도구: {name}")


# 서버 실행
async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

---

## 단계 6: 테스트하기

`tests/test_rag_server.py` 생성:

```python
"""
RAG 서버 테스트
"""

import asyncio
from mcp.client import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_rag_server():
    """RAG 서버 전체 기능 테스트"""
    
    server_params = StdioServerParameters(
        command="python",
        args=["servers/rag_server.py"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            print("=" * 70)
            print("📚 RAG 서버 테스트")
            print("=" * 70)
            
            # 1. 샘플 문서들 업로드
            print("\n1️⃣ 문서 업로드:")
            
            documents = [
                {
                    "doc_id": "doc_python_001",
                    "title": "Python 기초",
                    "content": "Python은 간결하고 읽기 쉬운 문법을 가진 프로그래밍 언어입니다. "
                               "동적 타이핑을 지원하며, 다양한 라이브러리가 있어 데이터 분석, "
                               "웹 개발, 머신러닝 등 여러 분야에서 사용됩니다."
                },
                {
                    "doc_id": "doc_mcp_001",
                    "title": "MCP란 무엇인가",
                    "content": "MCP(Model Context Protocol)는 AI 모델이 외부 도구와 "
                               "데이터에 접근할 수 있게 해주는 표준 프로토콜입니다. "
                               "Claude와 같은 AI가 파일 시스템, 데이터베이스, API 등을 "
                               "일관된 방식으로 사용할 수 있게 합니다."
                },
                {
                    "doc_id": "doc_rag_001",
                    "title": "RAG 시스템 이해하기",
                    "content": "RAG(Retrieval Augmented Generation)는 검색과 생성을 "
                               "결합한 방식입니다. 먼저 관련 문서를 검색한 후, 그 문서를 "
                               "바탕으로 답변을 생성합니다. 이를 통해 AI가 최신 정보나 "
                               "특정 도메인 지식을 활용할 수 있습니다."
                }
            ]
            
            for doc in documents:
                result = await session.call_tool("upload_document", doc)
                print(result.content[0].text)
                print("-" * 70)
            
            # 2. 문서 목록 확인
            print("\n2️⃣ 전체 문서 목록:")
            result = await session.call_tool("list_documents", {})
            print(result.content[0].text)
            
            # 3. 검색 테스트 1
            print("\n3️⃣ 검색: 'AI와 프로토콜'")
            result = await session.call_tool(
                "search_documents",
                {"query": "AI와 프로토콜", "top_k": 2}
            )
            print(result.content[0].text)
            
            # 4. 검색 테스트 2
            print("\n4️⃣ 검색: 'Python 프로그래밍'")
            result = await session.call_tool(
                "search_documents",
                {"query": "Python 프로그래밍", "top_k": 2}
            )
            print(result.content[0].text)
            
            # 5. 검색 테스트 3
            print("\n5️⃣ 검색: '문서 검색 방법'")
            result = await session.call_tool(
                "search_documents",
                {"query": "문서 검색 방법", "top_k": 2}
            )
            print(result.content[0].text)
            
            print("\n" + "=" * 70)
            print("✅ RAG 서버 테스트 완료!")
            print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_rag_server())
```

### 테스트 실행

```bash
python tests/test_rag_server.py
```

---

## 단계 7: Claude와 연결

`claude_desktop_config.json`에 추가:

```json
{
  "mcpServers": {
    "rag": {
      "command": "python",
      "args": [
        "C:\\Users\\[사용자이름]\\mcp-learning\\servers\\rag_server.py"
      ]
    }
  }
}
```

Claude Desktop 재시작 후 테스트!

---

## Claude와 RAG 사용 예시

### 예시 1: 문서 업로드 후 검색

```
User: "MCP에 대한 문서를 하나 저장해줘. 
      ID는 doc_mcp_basic으로 하고, 
      제목은 'MCP 기초 개념'으로 해줘."

Claude: [upload_document 도구 사용]

User: "지금 저장한 문서랑 비슷한 내용의 문서를 찾아줘"

Claude: [search_documents 도구 사용]
```

### 예시 2: 대화 기반 학습

```
User: "내가 Python, MCP, RAG에 대한 공부 내용을 
      각각 문서로 저장하고 싶은데 도와줄래?"

Claude: "네! 각 주제별로 어떤 내용을 저장하고 싶으신가요?"

User: [내용 설명]

Claude: [각 주제별로 upload_document 실행]

User: "RAG와 관련된 내용 찾아줘"

Claude: [search_documents로 RAG 관련 문서 검색]
```

---

## 성능 개선 아이디어

### 1. 청크 분할
긴 문서는 작은 청크로 나눠서 저장:

```python
def split_into_chunks(text: str, chunk_size: int = 500) -> List[str]:
    """텍스트를 청크로 분할"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks
```

### 2. 하이브리드 검색
키워드 + 의미 검색 결합:

```python
def hybrid_search(query: str, top_k: int = 5):
    """키워드 + 임베딩 검색"""
    # 1. 임베딩 검색
    semantic_results = search_similar_documents(query, top_k * 2)
    
    # 2. 키워드 검색
    keyword_results = keyword_search(query)
    
    # 3. 결과 병합 및 재정렬
    combined = merge_results(semantic_results, keyword_results)
    
    return combined[:top_k]
```

### 3. 메타데이터 필터링
날짜, 카테고리 등으로 필터링:

```python
Tool(
    name="search_documents_filtered",
    description="필터를 적용한 문서 검색",
    inputSchema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "category": {"type": "string"},
            "date_from": {"type": "string"},
            "date_to": {"type": "string"}
        }
    }
)
```

---

## 연습 문제

### 문제 1: 문서 업데이트 기능

기존 문서의 내용을 업데이트하는 `update_document` 도구를 만들어보세요.

**힌트:**
1. 기존 문서 찾기
2. 내용 업데이트
3. 새로운 임베딩 생성
4. 인덱스 저장

### 문제 2: 문서 통계

저장된 문서들의 통계를 보여주는 `get_statistics` 도구를 만들어보세요.

**포함할 정보:**
- 전체 문서 수
- 평균 문서 길이
- 가장 최근/오래된 문서

### 문제 3: 배치 업로드

여러 문서를 한 번에 업로드하는 기능을 추가해보세요.

---

## LangGraph vs MCP RAG 비교

### LangGraph 방식
```python
# 노드마다 로직이 포함됨
def retrieve_node(state):
    query = state["query"]
    docs = vector_store.search(query)  # 직접 구현
    return {"documents": docs}

def generate_node(state):
    docs = state["documents"]
    answer = llm.invoke(docs)
    return {"answer": answer}

graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
```

**문제점:**
- 검색 로직이 그래프에 종속
- 다른 프로젝트에서 재사용 어려움
- 새 데이터 소스 추가 시 전체 수정 필요

### MCP 방식
```python
# MCP 서버 (독립적)
@app.call_tool()
async def call_tool(name, arguments):
    if name == "search_documents":
        return search_documents(arguments["query"])

# Claude가 자동으로 사용
# 별도 그래프 구성 불필요!
```

**장점:**
- ✅ 검색 로직이 독립적인 서버
- ✅ 어떤 프로젝트에서든 재사용
- ✅ Claude가 상황에 맞게 도구 선택
- ✅ 새 기능 추가 시 서버만 수정

---

## 실전 배포 고려사항

### 1. 확장성
- 문서가 많아지면 JSON → 실제 Vector DB (Pinecone, Weaviate)
- 임베딩 모델 → API (OpenAI, Cohere)

### 2. 보안
- API 키 환경 변수로 관리
- 사용자 인증 추가
- 민감한 문서 암호화

### 3. 성능
- 캐싱 추가
- 비동기 처리 최적화
- 배치 처리

---

## 축하합니다! 🎉

여러분은 이제:
- ✅ MCP의 모든 핵심 개념을 이해했습니다
- ✅ 실용적인 MCP 서버를 만들 수 있습니다
- ✅ RAG 시스템을 MCP로 구현할 수 있습니다
- ✅ Claude와 통합할 수 있습니다

---

## 다음 학습 방향

### 1. 공식 MCP 문서
- [MCP 공식 사이트](https://modelcontextprotocol.io/)
- [Python SDK 문서](https://github.com/modelcontextprotocol/python-sdk)

### 2. 고급 주제
- 여러 MCP 서버 조합
- 스트리밍 응답 구현
- 에러 핸들링 고도화
- 프로덕션 배포

### 3. 커뮤니티
- GitHub에서 MCP 예제 둘러보기
- Discord 커뮤니티 참여
- 자신만의 MCP 서버 공유

---

## 마무리

MCP 학습을 완료하신 것을 축하드립니다!

이제 여러분은:
- AI 도구를 표준화된 방식으로 만들 수 있고
- 재사용 가능한 서비스를 구축할 수 있으며
- Claude와 같은 AI를 더 강력하게 만들 수 있습니다

**계속 배우고, 만들고, 공유하세요!** 🚀

[01-MCP(Model Context Protocol)의 개요](https://lee-keonhee.github.io/zero_to_deep_kh/blog/MCP_1/#/)
[02-MCP(Model Context Protocol) 서버 만들기](https://lee-keonhee.github.io/zero_to_deep_kh/blog/MCP_2/#/)
[03-MCP(Model Context Protocol) 도구 추가하기](https://lee-keonhee.github.io/zero_to_deep_kh/blog/MCP_3/#/)
[04-MCP(Model Context Protocol) LLM 연결하기](https://lee-keonhee.github.io/zero_to_deep_kh/blog/MCP_4/#/)
[06-실전-프로젝트-RAG](https://lee-keonhee.github.io/zero_to_deep_kh/blog/MCP_5/#/)
