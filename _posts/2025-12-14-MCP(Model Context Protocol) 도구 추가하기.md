---
layout: post
title: MCP(Model Context Protocol) 도구 추가하기
summary: MCP(Model Context Protocol) 도구 추가하기
author: keonhee
date: 2025-12-14 10:00:00 +0900
category: MCP
keywords: MCP
permalink: /blog/MCP_3/
usemathjax: true
thumbnail: /assets/img/posts/overview_of_NLP_1.png
---
# 03. 도구 추가하기: 실용적인 기능 만들기

#### 목표

계산기를 넘어서 실제로 유용한 도구들을 MCP 서버에 추가해봅니다.

**만들 것:**

- 파일 읽기/쓰기 도구
- 메모 저장/검색 도구
- JSON 데이터 처리 도구

---

#### 프로젝트: 메모 관리 서버

LangGraph로 RAG에서 "문서를 직접 관리하는 MCP 서버"를 만들어볼 거예요!

###### 서버 기능 설계

```
📝 Memo Server
├── save_memo    - 메모 저장
├── get_memo     - 특정 메모 가져오기
├── list_memos   - 모든 메모 목록 보기
└── search_memos - 키워드로 메모 검색
```

---

#### 단계 1: 기본 구조 만들기

`servers/memo_server.py` 파일 생성:

```python
"""
메모 관리 MCP 서버
메모를 저장하고 검색하는 기능을 제공합니다.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio

## 서버 인스턴스
app = Server("memo-server")

## 메모 저장 경로
MEMO_DIR = Path("memos")
MEMO_DIR.mkdir(exist_ok=True)  ## 폴더가 없으면 생성


def get_memo_path(memo_id: str) -> Path:
    """메모 파일 경로 반환"""
    return MEMO_DIR / f"{memo_id}.json"


def save_memo_to_file(memo_id: str, title: str, content: str):
    """메모를 JSON 파일로 저장"""
    memo_data = {
        "id": memo_id,
        "title": title,
        "content": content,
        "created_at": datetime.now().isoformat()
    }
    
    filepath = get_memo_path(memo_id)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(memo_data, f, ensure_ascii=False, indent=2)
    
    return memo_data


def load_memo_from_file(memo_id: str):
    """파일에서 메모 읽기"""
    filepath = get_memo_path(memo_id)
    
    if not filepath.exists():
        return None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def list_all_memos():
    """모든 메모 목록 가져오기"""
    memos = []
    for filepath in MEMO_DIR.glob("*.json"):
        with open(filepath, 'r', encoding='utf-8') as f:
            memo = json.load(f)
            memos.append({
                "id": memo["id"],
                "title": memo["title"],
                "created_at": memo["created_at"]
            })
    return memos


def search_memos_by_keyword(keyword: str):
    """키워드로 메모 검색"""
    results = []
    keyword_lower = keyword.lower()
    
    for filepath in MEMO_DIR.glob("*.json"):
        with open(filepath, 'r', encoding='utf-8') as f:
            memo = json.load(f)
            
            ## 제목이나 내용에 키워드가 있으면 추가
            if (keyword_lower in memo["title"].lower() or 
                keyword_lower in memo["content"].lower()):
                results.append(memo)
    
    return results


## 이제 MCP 도구들을 정의합니다...
```

---

#### 단계 2: 도구 목록 정의하기

같은 파일에 계속 작성:

```python
@app.list_tools()
async def list_tools() -> list[Tool]:
    """제공하는 도구 목록"""
    return [
        ## 1. 메모 저장
        Tool(
            name="save_memo",
            description="새로운 메모를 저장합니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "memo_id": {
                        "type": "string",
                        "description": "메모의 고유 ID (예: memo_001)"
                    },
                    "title": {
                        "type": "string",
                        "description": "메모 제목"
                    },
                    "content": {
                        "type": "string",
                        "description": "메모 내용"
                    }
                },
                "required": ["memo_id", "title", "content"]
            }
        ),
        
        ## 2. 메모 가져오기
        Tool(
            name="get_memo",
            description="특정 ID의 메모를 가져옵니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "memo_id": {
                        "type": "string",
                        "description": "가져올 메모의 ID"
                    }
                },
                "required": ["memo_id"]
            }
        ),
        
        ## 3. 메모 목록
        Tool(
            name="list_memos",
            description="저장된 모든 메모의 목록을 반환합니다",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        
        ## 4. 메모 검색
        Tool(
            name="search_memos",
            description="키워드로 메모를 검색합니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "검색할 키워드"
                    }
                },
                "required": ["keyword"]
            }
        )
    ]
```

---

#### 단계 3: 도구 실행 로직 구현

```python
@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """도구 실행"""
    
    if name == "save_memo":
        ## 메모 저장
        memo_id = arguments["memo_id"]
        title = arguments["title"]
        content = arguments["content"]
        
        memo = save_memo_to_file(memo_id, title, content)
        
        return [
            TextContent(
                type="text",
                text=f"✅ 메모 저장 완료!\n"
                     f"ID: {memo['id']}\n"
                     f"제목: {memo['title']}\n"
                     f"생성 시간: {memo['created_at']}"
            )
        ]
    
    elif name == "get_memo":
        ## 메모 가져오기
        memo_id = arguments["memo_id"]
        memo = load_memo_from_file(memo_id)
        
        if memo is None:
            return [
                TextContent(
                    type="text",
                    text=f"❌ 메모를 찾을 수 없습니다: {memo_id}"
                )
            ]
        
        return [
            TextContent(
                type="text",
                text=f"📝 {memo['title']}\n"
                     f"생성: {memo['created_at']}\n"
                     f"\n{memo['content']}"
            )
        ]
    
    elif name == "list_memos":
        ## 메모 목록
        memos = list_all_memos()
        
        if not memos:
            return [
                TextContent(
                    type="text",
                    text="📭 저장된 메모가 없습니다."
                )
            ]
        
        memo_list = "\n".join([
            f"- {m['id']}: {m['title']} (생성: {m['created_at']})"
            for m in memos
        ])
        
        return [
            TextContent(
                type="text",
                text=f"📋 메모 목록 ({len(memos)}개):\n{memo_list}"
            )
        ]
    
    elif name == "search_memos":
        ## 메모 검색
        keyword = arguments["keyword"]
        results = search_memos_by_keyword(keyword)
        
        if not results:
            return [
                TextContent(
                    type="text",
                    text=f"🔍 '{keyword}'에 대한 검색 결과가 없습니다."
                )
            ]
        
        result_text = "\n\n".join([
            f"📝 {m['title']} (ID: {m['id']})\n{m['content'][:100]}..."
            for m in results
        ])
        
        return [
            TextContent(
                type="text",
                text=f"🔍 '{keyword}' 검색 결과 ({len(results)}개):\n\n{result_text}"
            )
        ]
    
    else:
        raise ValueError(f"알 수 없는 도구: {name}")


## 서버 실행 코드
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

#### 단계 4: 테스트 스크립트 작성

`tests/test_memo_server.py` 생성:

```python
"""
메모 서버 테스트
"""

import asyncio
from mcp.client import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_memo_server():
    """메모 서버 전체 기능 테스트"""
    
    server_params = StdioServerParameters(
        command="python",
        args=["servers/memo_server.py"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            print("=" * 60)
            print("📝 메모 서버 테스트 시작")
            print("=" * 60)
            
            ## 1. 도구 목록 확인
            print("\n1️⃣ 사용 가능한 도구:")
            tools = await session.list_tools()
            for tool in tools.tools:
                print(f"   - {tool.name}: {tool.description}")
            
            ## 2. 메모 저장
            print("\n2️⃣ 메모 저장 테스트:")
            result = await session.call_tool(
                "save_memo",
                {
                    "memo_id": "memo_001",
                    "title": "MCP 학습 노트",
                    "content": "MCP는 Model Context Protocol의 약자다. "
                               "AI가 외부 도구를 사용할 수 있게 해주는 표준 프로토콜이다."
                }
            )
            print(result.content[0].text)
            
            ## 3. 또 다른 메모 저장
            print("\n3️⃣ 두 번째 메모 저장:")
            result = await session.call_tool(
                "save_memo",
                {
                    "memo_id": "memo_002",
                    "title": "Python 팁",
                    "content": "async/await는 비동기 프로그래밍을 위한 키워드다."
                }
            )
            print(result.content[0].text)
            
            ## 4. 메모 목록 확인
            print("\n4️⃣ 전체 메모 목록:")
            result = await session.call_tool("list_memos", {})
            print(result.content[0].text)
            
            ## 5. 특정 메모 가져오기
            print("\n5️⃣ 특정 메모 조회 (memo_001):")
            result = await session.call_tool(
                "get_memo",
                {"memo_id": "memo_001"}
            )
            print(result.content[0].text)
            
            ## 6. 키워드 검색
            print("\n6️⃣ 키워드 검색 ('MCP'):")
            result = await session.call_tool(
                "search_memos",
                {"keyword": "MCP"}
            )
            print(result.content[0].text)
            
            ## 7. 없는 메모 조회
            print("\n7️⃣ 존재하지 않는 메모 조회:")
            result = await session.call_tool(
                "get_memo",
                {"memo_id": "memo_999"}
            )
            print(result.content[0].text)
            
            print("\n" + "=" * 60)
            print("✅ 모든 테스트 완료!")
            print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_memo_server())
```

---

#### 실행해보기

```bash
## 테스트 실행
python tests/test_memo_server.py
```

실행 후 `memos/` 폴더를 확인해보세요! JSON 파일들이 생성되어 있을 거예요.

---

#### 코드 분석: 핵심 개념

###### 1. 파일 시스템 사용

```python
MEMO_DIR = Path("memos")
MEMO_DIR.mkdir(exist_ok=True)
```

- `Path`: 경로를 다루는 객체 (os.path보다 편리)
- `mkdir(exist_ok=True)`: 폴더가 없으면 생성, 있으면 무시

###### 2. JSON으로 데이터 저장

```python
json.dump(memo_data, f, ensure_ascii=False, indent=2)
```

- `ensure_ascii=False`: 한글 깨짐 방지
- `indent=2`: 읽기 쉽게 들여쓰기

###### 3. glob으로 파일 검색

```python
for filepath in MEMO_DIR.glob("*.json"):
```

- `glob("*.json")`: .json으로 끝나는 모든 파일 찾기
- 리스트가 아닌 제너레이터 반환 (메모리 효율적)

---

#### 연습 문제

###### 문제 1: 메모 삭제 기능 추가

`delete_memo` 도구를 추가해보세요.

**요구사항:**

- 메모 ID를 입력받아 해당 파일 삭제
- 파일이 없으면 에러 메시지 반환
- 삭제 성공 시 확인 메시지 반환

<details> <summary>힌트 보기</summary>

```python
## 도구 정의
Tool(
    name="delete_memo",
    description="메모를 삭제합니다",
    inputSchema={
        "type": "object",
        "properties": {
            "memo_id": {"type": "string", "description": "삭제할 메모 ID"}
        },
        "required": ["memo_id"]
    }
)

## 실행 로직
elif name == "delete_memo":
    memo_id = arguments["memo_id"]
    filepath = get_memo_path(memo_id)
    
    if not filepath.exists():
        return [TextContent(type="text", text=f"❌ 메모를 찾을 수 없습니다: {memo_id}")]
    
    filepath.unlink()  ## 파일 삭제
    return [TextContent(type="text", text=f"🗑️ 메모 삭제 완료: {memo_id}")]
```

</details>

###### 문제 2: 메모 수정 기능

`update_memo` 도구를 만들어서 기존 메모의 내용을 수정할 수 있게 해보세요.

**도전 과제:** 수정 시 `updated_at` 필드도 추가해보세요!

---

#### 실전 응용: RAG와 연결하기

여러분이 LangGraph로 만든 RAG 시스템을 떠올려보세요.

**기존 방식:**

```python
## LangGraph 노드 안에 직접 구현
def retrieve_documents(query):
    ## 문서 검색 로직
    return docs
```

**MCP 방식:**

```python
## MCP 서버로 분리
## servers/document_server.py

@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="search_documents",
            description="문서를 검색합니다",
            inputSchema={...}
        )
    ]

## Claude가 필요할 때 자동으로 호출!
```

**장점이 보이시나요?**

- 문서 검색 로직이 독립적인 서버로 분리
- LangGraph 없이도 다른 프로젝트에서 재사용 가능
- Claude가 직접 서버에 접근해서 문서 검색

---

#### 디버깅 체크리스트

###### ✅ 파일이 생성되지 않는다

- `MEMO_DIR.mkdir(exist_ok=True)` 실행되었는지 확인
- 파일 쓰기 권한 확인
- `with open()` 블록이 제대로 닫혔는지 확인

###### ✅ JSON 한글이 깨진다

- `ensure_ascii=False` 옵션 추가했는지 확인
- 파일 열 때 `encoding='utf-8'` 지정했는지 확인

###### ✅ 메모를 찾을 수 없다

- `memo_id`가 정확한지 확인
- 파일 경로 로직 검증: `print(get_memo_path(memo_id))`

---

#### 이해도 확인

다음 질문에 답할 수 있나요?

1. **왜 헬퍼 함수들을 따로 만들었나요?**
    
    - 힌트: `save_memo_to_file`, `load_memo_from_file` 등
2. **inputSchema에서 `required` 필드의 역할은?**
    
3. **이 메모 서버를 다른 프로젝트에서 사용하려면?**
    
    - 힌트: 서버만 실행하면...
4. **LangGraph 방식과 MCP 방식의 가장 큰 차이는?**
    

---

#### 다음 단계

축하합니다! 실용적인 MCP 서버를 만들었어요! 🎉

이제 가장 중요한 단계가 남았습니다: **LLM모델과 실제로 연결해보기!**

**다음: [04-LLM 연결하기](https://lee-keonhee.github.io/zero_to_deep_kh/blog/MCP_4/##/)**

