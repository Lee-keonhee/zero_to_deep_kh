---
layout: post
title: MCP(Model Context Protocol) 서버 만들기
summary: MCP(Model Context Protocol) 서버 만들기
author: keonhee
date: 2025-12-13 10:00:00 +0900
category: MCP
keywords: MCP
permalink: /blog/MCP_2/
usemathjax: true
thumbnail: /assets/img/posts/overview_of_NLP_1.png
---
# 02. 환경 설정 및 서버 생성

## 목표

Windows 환경에서 Python 3.12와 PyCharm을 사용해 MCP 개발 환경을 구축해, MCP 서버를 만들고 기본 구조를 이해합니다.

---

## 사전 확인사항

### 1. Python 버전 확인

```bash
python --version
```

**결과:** `Python 3.12.x` 가 나와야 합니다.

만약 다른 버전이 나온다면:

- Python 3.12를 [공식 홈페이지](https://www.python.org/downloads/)에서 다운로드
- 설치 시 "Add Python to PATH" 체크 필수!

### 2. pip 업데이트

```bash
python -m pip install --upgrade pip
```

---

## MCP Python SDK 설치

### 1. 프로젝트 폴더 생성

### 2. MCP SDK 설치

PyCharm 터미널에서 실행:

```bash
pip install mcp
```

**설치 확인:**

```bash
pip list | findstr mcp
```

다음과 같은 결과가 나와야 합니다:

```python
mcp                    1.24.0    # x.x.x
```

---

## 프로젝트 구조 준비

### 폴더 구조 만들기

프로젝트 루트에 다음과 같은 구조를 만들어주세요:

```
mcp-learning/
├── servers/          # MCP 서버 파일들
│   └── __init__.py
├── tests/            # 테스트 코드
│   └── __init__.py
└── README.md
```


---

## 추가 패키지 설치

앞으로 필요할 패키지들을 미리 설치해봅시다:

```bash
pip install pydantic httpx
```

**각 패키지의 역할:**

- `pydantic`: 데이터 검증 (MCP에서 자동으로 사용)
- `httpx`: HTTP 클라이언트 (나중에 외부 API 연결 시 사용)

---

## 설치 확인하기

### 테스트 스크립트 실행

`servers/` 폴더에 `test_install.py` 파일 생성:

```python
"""MCP 설치 확인 스크립트"""

def check_imports():
    """필수 패키지 import 테스트"""
    try:
        import mcp
        print("✓ mcp 패키지 import 성공")
        
        import pydantic
        print("✓ pydantic 패키지 import 성공")
        print(f"  버전: {pydantic.__version__}")
        
        import httpx
        print("✓ httpx 패키지 import 성공")
        print(f"  버전: {httpx.__version__}")
        
        print("\n🎉 모든 패키지가 정상적으로 설치되었습니다!")
        return True
        
    except ImportError as e:
        print(f"❌ 오류 발생: {e}")
        print("pip install mcp pydantic httpx 를 다시 실행해보세요.")
        return False

if __name__ == "__main__":
    check_imports()
```

**실행 방법:**

1. PyCharm에서 `test_install.py` 파일 열기
2. 우클릭 → `Run 'test_install'` 또는 `Shift + F10`

**예상 결과:**

```
✓ mcp 패키지 import 성공
✓ pydantic 패키지 import 성공
  버전: x.x.x
✓ httpx 패키지 import 성공
  버전: x.x.x

🎉 모든 패키지가 정상적으로 설치되었습니다!
```

---
# MCP 서버 만들기

**만들 것:** 두 숫자를 더하는 계산기 서버


## MCP 서버의 기본 구조 이해하기

MCP 서버는 크게 3가지로 구성됩니다:

```
📦 MCP Server
├── 📋 도구 목록 (Tools) - 서버가 제공하는 기능들
├── 🔧 도구 실행 (Tool Execution) - 실제 기능 수행
└── 🚀 서버 실행 (Server Run) - 서버를 켜는 부분
```

---

## 단계별로 만들어보기

### 1단계: 기본 템플릿 작성

`servers/calculator_server.py` 파일을 생성하고 다음 코드를 작성하세요:

```python
"""
간단한 계산기 MCP 서버
두 숫자를 더하는 기능을 제공합니다.
"""

from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio

# 1. 서버 인스턴스 생성
app = Server("calculator-server")


# 2. 제공할 도구 목록 정의
@app.list_tools()
async def list_tools() -> list[Tool]:
    """
    이 서버가 제공하는 도구들을 알려줍니다.
    클라이언트(Claude)가 "어떤 기능들이 있어?" 라고 물으면
    이 함수가 실행됩니다.
    """
    return [
        Tool(
            name="add",
            description="두 숫자를 더합니다",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "첫 번째 숫자"
                    },
                    "b": {
                        "type": "number",
                        "description": "두 번째 숫자"
                    }
                },
                "required": ["a", "b"]
            }
        )
    ]


# 3. 도구 실행 로직
@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """
    클라이언트가 도구를 실제로 사용할 때 실행됩니다.
    예: Claude가 "5 + 3을 계산해줘" 라고 하면 이 함수가 실행
    """
    if name == "add":
        # 인자 가져오기
        a = arguments["a"]
        b = arguments["b"]
        
        # 계산 수행
        result = a + b
        
        # 결과 반환
        return [
            TextContent(
                type="text",
                text=f"{a} + {b} = {result}"
            )
        ]
    else:
        raise ValueError(f"알 수 없는 도구: {name}")


# 4. 서버 실행
async def main():
    """서버를 시작합니다"""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


# 진입점
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

---

## 코드 설명: 한 줄씩 이해하기

### 1. 서버 생성

```python
app = Server("calculator-server")
```

- `Server` 클래스로 서버 인스턴스 생성
- `"calculator-server"` = 이 서버의 이름

### 2. 도구 목록 (@app.list_tools)

```python
@app.list_tools()
async def list_tools() -> list[Tool]:
```

- `@app.list_tools()` = 데코레이터, 이 함수를 "도구 목록 제공자"로 등록
- Claude가 "무슨 기능 있어?" 하면 이 함수가 실행됨

**Tool 객체 구조:**

```python
Tool(
    name="add",              # 도구 이름
    description="설명",      # 도구가 뭐하는지
    inputSchema={...}        # 필요한 입력값 정의
)
```

### 3. inputSchema (중요!)

```python
"inputSchema": {
    "type": "object",
    "properties": {
        "a": {"type": "number", "description": "첫 번째 숫자"},
        "b": {"type": "number", "description": "두 번째 숫자"}
    },
    "required": ["a", "b"]
}
```

이건 **JSON Schema** 형식이에요. Claude에게:

- "a와 b 두 개의 숫자가 필요해"
- "둘 다 필수야" 라고 알려주는 거죠.

### 4. 도구 실행 (@app.call_tool)

```python
@app.call_tool()
async def call_tool(name: str, arguments: dict):
```

- Claude가 실제로 도구를 **사용**할 때 실행
- `name`: 어떤 도구를 쓸지 ("add")
- `arguments`: 입력값들 ({"a": 5, "b": 3})

### 5. 결과 반환

```python
return [
    TextContent(
        type="text",
        text=f"{a} + {b} = {result}"
    )
]
```

- 결과를 `TextContent` 형태로 반환
- 리스트로 감싸는 이유: 여러 개의 결과를 반환할 수도 있어서

---

## 서버 실행해보기

### 터미널에서 직접 실행

PyCharm 터미널에서:

```bash
python servers/calculator_server.py
```

**예상 출력:** 서버가 시작되면 아무것도 출력되지 않고 **대기 상태**가 됩니다. 이게 정상이에요! 서버는 클라이언트의 요청을 기다리고 있는 거예요.

**종료 방법:** `Ctrl + C`

---

## 서버 테스트하기

이제 서버가 제대로 작동하는지 테스트해봅시다!

### 테스트 스크립트 작성

`tests/test_calculator.py` 파일 생성:

```python
"""
계산기 서버 테스트
서버가 제대로 작동하는지 확인합니다.
"""

import asyncio
from mcp.client import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_calculator():
    """계산기 서버 테스트"""
    
    # 서버 실행 파라미터
    server_params = StdioServerParameters(
        command="python",
        args=["servers/calculator_server.py"]
    )
    
    # 서버에 연결
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            
            # 초기화
            await session.initialize()
            
            # 1. 도구 목록 가져오기
            tools = await session.list_tools()
            print("📋 사용 가능한 도구:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")
            
            print("\n" + "="*50 + "\n")
            
            # 2. add 도구 사용하기
            print("🧮 계산 실행: 5 + 3")
            result = await session.call_tool("add", {"a": 5, "b": 3})
            print(f"결과: {result.content[0].text}")
            
            print("\n" + "="*50 + "\n")
            
            # 3. 다른 숫자로 테스트
            print("🧮 계산 실행: 100 + 234")
            result = await session.call_tool("add", {"a": 100, "b": 234})
            print(f"결과: {result.content[0].text}")


if __name__ == "__main__":
    asyncio.run(test_calculator())
```

### 테스트 실행

```bash
python tests/test_calculator.py
```

**예상 출력:**

```
📋 사용 가능한 도구:
  - add: 두 숫자를 더합니다

==================================================

🧮 계산 실행: 5 + 3
결과: 5 + 3 = 8

==================================================

🧮 계산 실행: 100 + 234
결과: 100 + 234 = 334
```

---

## 연습 문제: 직접 해보세요!

테스트가 성공했다면, 이제 직접 수정해볼 시간입니다!

### 문제 1: 빼기 기능 추가하기

`calculator_server.py`를 수정해서 빼기 기능을 추가해보세요.

**힌트:**

1. `list_tools()` 함수에 새 Tool 추가
2. `call_tool()` 함수에 "subtract" 케이스 추가

<details> <summary>정답 보기 (클릭)</summary>


```python
# list_tools() 함수의 return에 추가:
Tool(
    name="subtract",
    description="두 숫자를 뺍니다",
    inputSchema={
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "첫 번째 숫자"},
            "b": {"type": "number", "description": "두 번째 숫자"}
        },
        "required": ["a", "b"]
    }
)

# call_tool() 함수에 추가:
elif name == "subtract":
    a = arguments["a"]
    b = arguments["b"]
    result = a - b
    return [
        TextContent(
            type="text",
            text=f"{a} - {b} = {result}"
        )
    ]
```


</details>

### 문제 2: 곱하기, 나누기도 추가해보기

위와 같은 방식으로 `multiply`와 `divide` 도구를 추가해보세요.

**주의:** 나누기는 0으로 나누는 경우를 처리해야 해요!

<details> <summary>나누기 에러 처리 힌트</summary>

```python
elif name == "divide":
    a = arguments["a"]
    b = arguments["b"]
    
    if b == 0:
        return [
            TextContent(
                type="text",
                text="오류: 0으로 나눌 수 없습니다"
            )
        ]
    
    result = a / b
    return [
        TextContent(
            type="text",
            text=f"{a} ÷ {b} = {result}"
        )
    ]
```

</details>

---

## 디버깅 팁

### 문제 1: "Server is already initialized"

**원인:** 서버가 이미 실행 중인데 다시 실행하려고 함 **해결:** 실행 중인 서버를 `Ctrl + C`로 종료 후 재시작

### 문제 2: "Tool not found"

**원인:** `call_tool()`에서 처리하지 않은 도구 이름 **해결:** `list_tools()`에 정의한 name과 `call_tool()`의 if문 확인

### 문제 3: "Missing required argument"

**원인:** inputSchema의 required 필드와 실제 arguments가 안 맞음 **해결:** 테스트할 때 필수 인자를 모두 전달했는지 확인

---

## 이해도 체크: 스스로 설명해보세요

다음 질문에 답할 수 있으면 다음 단계로 넘어갈 준비가 된 거예요:

1. **`@app.list_tools()` 데코레이터는 언제 실행되나요?**
	    클라이언트가 서버와 연결된 직후, 클라이언트가 도구 정보를 요청할 때
2. **inputSchema는 왜 필요한가요?**
	    LLM이 도구를 올바르게 사용하도록 유도하는 **사용설명서** 역할을 하기때문에, LLM은 inputSchema를 보고 정의된 형식에 맞춰 도구 호출 인자를 JSON형태로 생성.
3. **`call_tool()` 함수의 arguments는 어디서 오나요?**
	    `call_tool()` 함수의 `arguments` 딕셔너리는 **LLM(대규모 언어 모델)이 생성하여 전송한 JSON 객체**입니다.
4. **TextContent로 감싸서 반환하는 이유는?**
	    **MCP 프로토콜의 표준 데이터 구조를 준수**하기 위함으로, `TextContent` 외에도 이미지 파일(예: `ImageContent`), 파일 자체(예: `FileContent`) 등 다양한 콘텐츠 타입을 반환할 수 있도록 설계되어 있습니다.

---

## 핵심 정리

```
MCP 서버 = 3가지 핵심 함수

1. list_tools()  → "내가 뭘 할 수 있는지" 알려줌
2. call_tool()   → "실제로 일을 처리함"
3. main()        → "서버를 실행함"
```

---

## 다음 단계

축하합니다! 첫 번째 MCP 서버를 만들었어요! 🎉

이제 더 복잡한 기능들을 추가해볼 거예요:

- 파일 읽기/쓰기
- 외부 API 호출
- 데이터베이스 연결

**다음: [03-도구-추가하기](https://lee-keonhee.github.io/zero_to_deep_kh/blog/MCP_3/#/)**

실제로 유용한 기능들을 서버에 추가해봅시다!