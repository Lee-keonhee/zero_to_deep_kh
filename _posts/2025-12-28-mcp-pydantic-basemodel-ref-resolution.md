---
layout: post
title: MCP에서 Pydantic BaseModel 사용 시 JSON Schema $ref 해소
summary: FastMCP와 Pydantic BaseModel을 사용할 때 발생하는 $ref 참조 문제와 LLM 통합을 위한 해결 방법
author: keonhee
date: 2025-12-28 23:30:00 +0900
category: MCP
keywords: MCP, Pydantic, BaseModel, JSON Schema, $ref, FastMCP, LLM
permalink: /blog/mcp-pydantic-basemodel-ref-resolution/
usemathjax: false
thumbnail: /assets/img/posts/mcp.png
imageNameKey: mcp
---


# MCP에서 Pydantic BaseModel 사용 시 주의사항

## 목차

1. [문제 상황]()
2. [JSON Schema 구조 이해하기]()
3. [원인 분석]()
4. [해결 방법]()
5. [대안 및 베스트 프랙티스]()

---

## 1. 문제 상황

### MCP 도구 정의 시 BaseModel 사용

```python
# weather_api.py
from pydantic import BaseModel, Field
from fastmcp import FastMCP

app = FastMCP("weather-api")

class WeatherQuery(BaseModel):
    city: str = Field(..., description="도시 이름")
    country: str = Field(default="KR", description="국가 코드")
    units: str = Field(default="metric", description="온도 단위")
    days: int = Field(default=7, description="예보 일수", ge=1, le=14)

@app.tool()
def get_weather_forecast(query: WeatherQuery):
    """날씨 예보 정보를 조회합니다."""
    # API 호출 로직
    pass
```

### 발생하는 JSON Schema 구조

FastMCP가 생성하는 `tool.inputSchema`:

```json
{
  "$defs": {
    "WeatherQuery": {
      "properties": {
        "city": {"type": "string", "description": "도시 이름"},
        "country": {"type": "string", "default": "KR", "description": "국가 코드"},
        "units": {"type": "string", "default": "metric", "description": "온도 단위"},
        "days": {"type": "integer", "default": 7, "description": "예보 일수", "minimum": 1, "maximum": 14}
      },
      "required": ["city"],
      "type": "object"
    }
  },
  "properties": {
    "query": {"$ref": "#/$defs/WeatherQuery"}
  },
  "required": ["query"],
  "type": "object"
}
```

### LLM이 받는 불완전한 정보

```python
# client.py에서 도구 정보 수집
self.available_tools.append({
    "server": server_name,
    "name": tool.name,
    "description": tool.description,
    "input_schema": tool.inputSchema,  # 전체 스키마 저장
})

# _get_tools_description에서 사용
props = tool['input_schema'].get('properties', {})
# → {"query": {"$ref": "#/$defs/WeatherQuery"}}
```

**결과:** LLM은 `$ref` 참조만 보고 `WeatherQuery`가 어떤 필드를 가지는지 전혀 알 수 없음 ❌

---

## 2. JSON Schema 구조 이해하기

### `$defs`와 `properties`의 역할 차이

JSON Schema는 두 가지 핵심 요소로 구성됩니다:

|요소|역할|비유|언제 생기나|
|---|---|---|---|
|**`$defs`**|재사용 가능한 타입 정의 저장소|📚 라이브러리|BaseModel 사용 시|
|**`properties`**|현재 객체의 실제 필드 선언|🎯 실제 파라미터 목록|항상|

### 전체 구조 예시

```python
{
    "$defs": {  # 📚 정의 저장소 (재사용용)
        "WeatherQuery": {
            "type": "object",
            "properties": {  # ← 여기도 properties!
                "city": {"type": "string", "description": "도시 이름"},
                "country": {"type": "string", "default": "KR"},
                "units": {"type": "string", "default": "metric"},
                "days": {"type": "integer", "default": 7}
            },
            "required": ["city"]
        }
    },
    "properties": {  # 🎯 이 스키마의 실제 파라미터
        "query": {"$ref": "#/$defs/WeatherQuery"}  # 참조로 연결
    },
    "required": ["query"]
}
```

### 왜 이런 구조를 사용하나?

**DRY 원칙 (Don't Repeat Yourself)**

```python
# 같은 Address 타입을 여러 곳에서 사용하는 경우
{
    "$defs": {
        "Address": {  # ✅ 한 번만 정의
            "properties": {
                "street": {"type": "string"},
                "city": {"type": "string"},
                "zipcode": {"type": "string"}
            }
        }
    },
    "properties": {
        "home_address": {"$ref": "#/$defs/Address"},  # 재사용
        "work_address": {"$ref": "#/$defs/Address"},  # 재사용
        "billing_address": {"$ref": "#/$defs/Address"}  # 재사용
    }
}

# ❌ $ref 없이 작성하면 중복 발생
{
    "properties": {
        "home_address": {
            "properties": {"street": ..., "city": ..., "zipcode": ...}  # 중복
        },
        "work_address": {
            "properties": {"street": ..., "city": ..., "zipcode": ...}  # 중복
        },
        "billing_address": {
            "properties": {"street": ..., "city": ..., "zipcode": ...}  # 중복
        }
    }
}
```

### BaseModel 사용 시 생성 과정

```python
# 1. Pydantic 모델 정의
class WeatherQuery(BaseModel):
    city: str
    country: str = "KR"

# 2. FastMCP 도구로 사용
@app.tool()
def get_weather(query: WeatherQuery):
    pass

# 3. FastMCP가 자동 생성하는 스키마
{
    "$defs": {
        "WeatherQuery": {  # ← 모델을 $defs에 정의
            "properties": {
                "city": {"type": "string"},
                "country": {"type": "string", "default": "KR"}
            }
        }
    },
    "properties": {
        "query": {"$ref": "#/$defs/WeatherQuery"}  # ← $ref로 참조
    }
}
```

**핵심:**

- `$defs`의 `WeatherQuery`: 실제 필드 정의 (`city`, `country`)
- `properties`의 `query`: 함수가 받는 파라미터 (1개: `query`)

### BaseModel을 안 쓰면?

```python
# 직접 파라미터 정의
@app.tool()
def get_weather(city: str, country: str = "KR"):
    pass

# 생성되는 스키마 (간단함!)
{
    "properties": {  # $defs 없음!
        "city": {"type": "string"},
        "country": {"type": "string", "default": "KR"}
    }
}
```

### 실제 코드에서의 차이

```python
# client.py - 도구 스키마 저장
self.available_tools.append({
    "input_schema": tool.inputSchema
})

# BaseModel 사용 시
tool.inputSchema = {
    "$defs": {"WeatherQuery": {...}},  # 정의는 여기
    "properties": {"query": {"$ref": ...}}  # 참조만 여기
}

# BaseModel 안 쓴 경우
tool.inputSchema = {
    "properties": {  # 바로 정의
        "city": {...},
        "country": {...}
    }
}
```

---

## 3. 원인 분석

### 문제가 발생하는 이유

```python
# _get_tools_description 메서드
props = tool['input_schema'].get('properties', {})
# → {"query": {"$ref": "#/$defs/WeatherQuery"}}

if props:
    tools_desc += f"- Parameters: {json.dumps(props, ensure_ascii=False)}\n"
    # → "- Parameters: {"query": {"$ref": "#/$defs/WeatherQuery"}}"
```

**문제:**

1. `properties`만 가져옴 → `$ref` 참조만 보임
2. `$defs`는 사용 안 함 → 실제 정의는 무시됨
3. LLM은 `$ref`를 해석 못함 → 어떤 필드가 필요한지 모름

### JSON Schema의 `$ref` 참조 메커니즘

`$ref`는 JSON Pointer 문법을 사용합니다:

```python
"$ref": "#/$defs/WeatherQuery"
#       │  │      └─ 참조할 정의 이름
#       │  └─ $defs 객체
#       └─ 현재 문서 루트
```

**해소 과정:**

```python
# 1. $ref 발견
{"$ref": "#/$defs/WeatherQuery"}

# 2. 경로 파싱
path = "#/$defs/WeatherQuery"
parts = path.split('/')  # ['#', '$defs', 'WeatherQuery']
target = parts[-1]  # 'WeatherQuery'

# 3. $defs에서 찾기
definitions = schema['$defs']
actual_schema = definitions[target]

# 4. 치환
{
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "country": {"type": "string", "default": "KR"},
        # ...
    }
}
```

### Pydantic의 동작 방식

Pydantic은 모든 BaseModel을 자동으로 `$defs`에 추출합니다:

```python
WeatherQuery.model_json_schema()

# 출력:
{
    "$defs": {"WeatherQuery": {...}},
    "properties": {...},
    "title": "WeatherQuery",
    "type": "object"
}
```

FastMCP, OpenAPI, JSON Schema 표준이 모두 이 방식을 사용합니다.

---

## 4. 해결 방법

### 4-1. `$ref` 해소 함수 구현 (권장 ✅)

**핵심 아이디어:** `$ref`를 만나면 `$defs`에서 실제 정의를 가져와 치환

```python
# client.py
def _resolve_schema_refs(self, schema: Any, definitions: Dict) -> Any:
    """JSON Schema의 $ref를 실제 스키마로 치환
    
    Args:
        schema: 해소할 스키마 (properties, items 등)
        definitions: $defs에서 가져온 정의 딕셔너리
    
    Returns:
        $ref가 모두 실제 내용으로 치환된 스키마
    """
    if isinstance(schema, dict):
        # $ref가 있으면 해당 정의로 교체
        if "$ref" in schema:
            # "#/$defs/WeatherQuery" → "WeatherQuery" 추출
            ref_path = schema["$ref"].split("/")[-1]
            if ref_path in definitions:
                # 재귀적으로 해소 (nested refs 처리)
                return self._resolve_schema_refs(definitions[ref_path], definitions)
        
        # 모든 키-값을 재귀적으로 처리
        return {k: self._resolve_schema_refs(v, definitions) 
                for k, v in schema.items()}
    
    elif isinstance(schema, list):
        # 배열의 각 요소 처리
        return [self._resolve_schema_refs(item, definitions) 
                for item in schema]
    
    # 기본 타입은 그대로 반환
    return schema
```

### 4-2. 처리 흐름 상세 설명

```python
# 입력 스키마
input_schema = {
    "$defs": {
        "WeatherQuery": {
            "properties": {
                "city": {"type": "string"},
                "country": {"type": "string", "default": "KR"}
            }
        }
    },
    "properties": {
        "query": {"$ref": "#/$defs/WeatherQuery"}
    }
}

# 1단계: 정의 추출
definitions = input_schema.get("$defs", {})
# → {"WeatherQuery": {...}}

# 2단계: properties 추출
properties = input_schema.get("properties", {})
# → {"query": {"$ref": "#/$defs/WeatherQuery"}}

# 3단계: $ref 해소
resolved = _resolve_schema_refs(properties, definitions)

# 처리 과정:
# 1) properties 순회
# 2) "query" 키 발견 → 값은 {"$ref": "#/$defs/WeatherQuery"}
# 3) "$ref" 키 발견!
# 4) "WeatherQuery" 추출
# 5) definitions["WeatherQuery"] 가져오기
# 6) 재귀 호출로 내부도 해소

# 결과
resolved = {
    "query": {
        "properties": {
            "city": {"type": "string"},
            "country": {"type": "string", "default": "KR"}
        }
    }
}
```

### 4-3. 중첩된 $ref 처리

```python
# 복잡한 경우: $ref 안에 또 $ref가 있는 경우
{
    "$defs": {
        "Address": {
            "properties": {
                "city": {"type": "string"},
                "zipcode": {"type": "string"}
            }
        },
        "Person": {
            "properties": {
                "name": {"type": "string"},
                "address": {"$ref": "#/$defs/Address"}  # ← 중첩 $ref
            }
        }
    },
    "properties": {
        "user": {"$ref": "#/$defs/Person"}  # ← 최상위 $ref
    }
}

# _resolve_schema_refs가 재귀적으로 처리
# 1) "user"의 $ref 해소 → Person 정의 가져옴
# 2) Person 안의 "address" $ref 해소 → Address 정의 가져옴
# 3) 최종 결과
{
    "user": {
        "properties": {
            "name": {"type": "string"},
            "address": {
                "properties": {
                    "city": {"type": "string"},
                    "zipcode": {"type": "string"}
                }
            }
        }
    }
}
```

### 4-4. MCP 도구 수집 시 적용

```python
async def _connect_mcp_server(self, server_name: str, command: str, args: List):
    """MCP 서버 연결 및 도구 수집 (스키마 해소 포함)"""
    server_params = StdioServerParameters(command=command, args=args)
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_response = await session.list_tools()

            for tool in tools_response.tools:
                # 1. $defs에서 정의 추출
                definitions = tool.inputSchema.get("$defs", {})
                
                # 2. properties에서 $ref 해소
                resolved_properties = self._resolve_schema_refs(
                    tool.inputSchema.get("properties", {}),
                    definitions
                )
                
                # 3. 해소된 스키마 저장
                self.available_tools.append({
                    "server": server_name,
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": {
                        "properties": resolved_properties,  # ✅ 해소된 스키마
                        "required": tool.inputSchema.get("required", [])
                    },
                })
                
                logger.info(f"✅ {tool.name} - Schema resolved")
```

### 4-5. 결과 비교

**Before (해소 전):**

```python
props = tool['input_schema'].get('properties', {})
# → {"query": {"$ref": "#/$defs/WeatherQuery"}}

# LLM이 받는 정보
"- Parameters: {"query": {"$ref": "#/$defs/WeatherQuery"}}"
```

**After (해소 후):**

```python
props = tool['input_schema'].get('properties', {})
# → 이미 해소된 상태로 저장되어 있음

# LLM이 받는 정보
"- Parameters: {
    "query": {
      "type": "object",
      "properties": {
        "city": {"type": "string", "description": "도시 이름"},
        "country": {"type": "string", "default": "KR", "description": "국가 코드"},
        "units": {"type": "string", "default": "metric"},
        "days": {"type": "integer", "default": 7, "minimum": 1, "maximum": 14}
      },
      "required": ["city"]
    }
  }"
```

---

## 5. 실전 구현

### 전체 클라이언트 코드

````python
import asyncio
from typing import Any, Dict, List
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

class MCPClient:
    """MCP 클라이언트 ($ref 해소 포함)"""
    
    def __init__(self, mcp_servers: Dict):
        self.available_tools = []
        self.loop = asyncio.new_event_loop()
        
        # MCP 서버 연결
        if mcp_servers:
            self.loop.run_until_complete(
                self._connect_all_servers(mcp_servers)
            )
    
    def _resolve_schema_refs(self, schema: Any, definitions: Dict) -> Any:
        """$ref 해소 - 재사용 가능한 유틸리티"""
        if isinstance(schema, dict):
            if "$ref" in schema:
                ref_path = schema["$ref"].split("/")[-1]
                if ref_path in definitions:
                    # 재귀적으로 해소
                    return self._resolve_schema_refs(
                        definitions[ref_path], 
                        definitions
                    )
            
            # 딕셔너리의 모든 값 처리
            return {
                k: self._resolve_schema_refs(v, definitions) 
                for k, v in schema.items()
            }
        
        elif isinstance(schema, list):
            # 리스트의 모든 요소 처리
            return [
                self._resolve_schema_refs(item, definitions) 
                for item in schema
            ]
        
        # 기본 타입은 그대로 반환
        return schema
    
    async def _connect_all_servers(self, mcp_servers: Dict):
        """모든 MCP 서버 동시 연결"""
        tasks = [
            self._connect_mcp_server(name, config["command"], config.get("args", []))
            for name, config in mcp_servers.items()
        ]
        await asyncio.gather(*tasks)
    
    async def _connect_mcp_server(self, server_name: str, command: str, args: List):
        """MCP 도구 등록 (스키마 해소 포함)"""
        print(f"🔌 MCP 서버 연결: {server_name}")
        
        server_params = StdioServerParameters(command=command, args=args)
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_response = await session.list_tools()

                for tool in tools_response.tools:
                    # 1. $defs 추출
                    definitions = tool.inputSchema.get("$defs", {})
                    
                    # 2. properties에서 $ref 해소
                    resolved_properties = self._resolve_schema_refs(
                        tool.inputSchema.get("properties", {}),
                        definitions
                    )
                    
                    # 3. 해소된 스키마 저장
                    self.available_tools.append({
                        "server": server_name,
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": {
                            "properties": resolved_properties,
                            "required": tool.inputSchema.get("required", [])
                        },
                    })
                    
                    print(f"  ✅ {tool.name}: {tool.description}")
    
    def get_tools_description(self) -> str:
        """LLM에게 제공할 도구 설명 생성"""
        if not self.available_tools:
            return ""
        
        tools_desc = "\n## Available Tools\n\n"
        for tool in self.available_tools:
            tools_desc += f"**{tool['name']}** (Server: {tool['server']})\n"
            tools_desc += f"- {tool['description']}\n"
            
            # 이제 완전히 해소된 스키마를 보여줌
            props = tool['input_schema'].get('properties', {})
            if props:
                import json
                tools_desc += f"- Parameters: {json.dumps(props, ensure_ascii=False, indent=2)}\n"
            tools_desc += "\n"
        
        tools_desc += """
## Tool Usage Instructions
If you need a tool, respond with ONLY this JSON format:
```json
{
    "tool_call": {
        "server": "server_name",
        "tool": "tool_name",
        "arguments": {"arg1": "value1"}
    }
}
````

""" return tools_desc

# 사용 예시

if **name** == "**main**": mcp_servers = { "weather-api": { "command": "python", "args": ["weather_api.py"] } }

```
client = MCPClient(mcp_servers)
print(client.get_tools_description())
```

```

### 로그 출력 예시

```

🔌 MCP 서버 연결: weather-api ✅ get_weather_forecast: 날씨 예보 정보를 조회합니다.

## Available Tools

**get_weather_forecast** (Server: weather-api)

- 날씨 예보 정보를 조회합니다.
- Parameters: { "query": { "type": "object", "properties": { "city": {"type": "string", "description": "도시 이름"}, "country": {"type": "string", "default": "KR", "description": "국가 코드"}, "units": {"type": "string", "default": "metric", "description": "온도 단위"}, "days": {"type": "integer", "default": 7, "minimum": 1, "maximum": 14} }, "required": ["city"] } }

````

### LLM이 생성하는 올바른 도구 호출

```json
{
  "tool_call": {
    "server": "weather-api",
    "tool": "get_weather_forecast",
    "arguments": {
      "city": "Seoul",
      "country": "KR",
      "units": "metric",
      "days": 7
    }
  }
}
````

---

## 6. 대안 및 베스트 프랙티스

### 대안 1: BaseModel 사용 안 함

```python
# Pydantic 모델 대신 직접 파라미터 정의
@app.tool()
def get_weather_forecast(
    city: str,
    country: str = "KR",
    units: str = "metric",
    days: int = 7
):
    """날씨 예보 조회"""
    pass

# 생성되는 스키마 (간단함!)
{
    "properties": {  # $defs 없음!
        "city": {"type": "string"},
        "country": {"type": "string", "default": "KR"},
        "units": {"type": "string", "default": "metric"},
        "days": {"type": "integer", "default": 7}
    }
}
```

**장점:**

- ✅ `$ref` 없이 바로 properties에 나옴
- ✅ 간단한 경우 빠른 구현
- ✅ `$ref` 해소 불필요

**단점:**

- ❌ 타입 검증 불가능 (Pydantic validation 없음)
- ❌ 기본값 처리 복잡
- ❌ 문서화 자동 생성 제한
- ❌ 여러 곳에서 재사용 시 중복 코드
- ❌ pattern, min/max 같은 고급 검증 불가

### 대안 2: Dict 타입 사용

```python
@app.tool()
def get_weather_forecast(params: dict):
    """날씨 예보 조회"""
    # 수동으로 검증 필요
    city = params.get("city")
    country = params.get("country", "KR")
    # ...
```

**문제점:**

- ❌ 타입 안전성 완전히 상실
- ❌ IDE 자동완성 불가능
- ❌ 런타임 에러 가능성 증가
- ❌ 수동 검증 코드 필요

### 권장 방식: BaseModel + `$ref` 해소 ✅

```python
# ========================================
# MCP 서버 정의 (weather_api.py)
# ========================================
class WeatherQuery(BaseModel):
    """날씨 조회 파라미터 - 타입 안전성 보장"""
    city: str = Field(..., description="도시 이름")
    country: str = Field(default="KR", description="국가 코드", pattern=r"^[A-Z]{2}$")
    units: str = Field(default="metric", description="온도 단위")
    days: int = Field(default=7, description="예보 일수", ge=1, le=14)
    
    @validator('city')
    def validate_city(cls, v):
        if len(v) < 2:
            raise ValueError('도시 이름은 최소 2자 이상이어야 합니다')
        return v

@app.tool()
def get_weather_forecast(query: WeatherQuery):
    """타입 검증된 안전한 도구"""
    # query는 자동으로 검증됨
    pass

# ========================================
# 클라이언트 구현 (client.py)
# ========================================
class MCPClient:
    def _resolve_schema_refs(self, schema, definitions):
        """$ref 해소 로직"""
        # ... (위에서 구현한 코드)
    
    async def _connect_mcp_server(self, ...):
        """도구 수집 시 스키마 해소"""
        definitions = tool.inputSchema.get("$defs", {})
        resolved_properties = self._resolve_schema_refs(
            tool.inputSchema.get("properties", {}),
            definitions
        )
        # 해소된 스키마 저장
```

**장점:**

- ✅ 타입 안전성 (Pydantic 검증)
- ✅ 기본값 처리 자동
- ✅ 문서화 자동 생성
- ✅ 스키마 재사용 가능
- ✅ LLM 통합 완벽 지원
- ✅ IDE 자동완성 지원
- ✅ 고급 검증 (pattern, validator 등)

---

## 핵심 정리

### ⚠️ 반드시 기억할 것

#### 1. JSON Schema 구조 이해

```python
{
    "$defs": {      # 📚 정의 저장소 (재사용용)
        "TypeName": {...}
    },
    "properties": { # 🎯 실제 파라미터 목록
        "param": {"$ref": "#/$defs/TypeName"}
    }
}
```

#### 2. 처리 순서

```python
# 항상 이 순서로 처리!
definitions = schema.get('$defs', {})      # 1. 정의 가져오기
properties = schema.get('properties', {})  # 2. 파라미터 가져오기
resolved = resolve_refs(properties, definitions)  # 3. $ref 해소
```

#### 3. BaseModel 사용 규칙

|상황|BaseModel 사용|`$ref` 해소 필요|
|---|---|---|
|간단한 도구 (5개 이하 파라미터)|선택사항|Yes (사용 시)|
|복잡한 도구 (6개 이상)|권장|**필수**|
|타입 검증 필요|**필수**|**필수**|
|스키마 재사용|**필수**|**필수**|

### ✅ 체크리스트

```python
# MCP 서버 개발 시
□ BaseModel 사용 시 $ref 생성 인지
□ FastMCP/Pydantic의 schema 구조 이해
□ $defs와 properties 역할 구분

# MCP 클라이언트 개발 시
□ _resolve_schema_refs() 함수 구현
□ 도구 수집 시 스키마 해소 적용
□ LLM에게 완전히 해소된 스키마 전달
□ 로그로 해소 결과 확인
□ 중첩된 $ref 처리 확인

# 테스트
□ LLM이 올바른 파라미터 생성하는지 확인
□ 중첩된 $ref도 정상 해소되는지 테스트
□ BaseModel의 validator가 작동하는지 확인
```

### 🎯 베스트 프랙티스

#### 1. 재사용 가능한 유틸리티 구현

```python
class SchemaResolver:
    """재사용 가능한 스키마 해소 클래스"""
    
    @staticmethod
    def resolve_refs(schema: Any, definitions: Dict) -> Any:
        """$ref 해소"""
        if isinstance(schema, dict):
            if "$ref" in schema:
                ref_path = schema["$ref"].split("/")[-1]
                if ref_path in definitions:
                    return SchemaResolver.resolve_refs(
                        definitions[ref_path],
                        definitions
                    )
            return {
                k: SchemaResolver.resolve_refs(v, definitions)
                for k, v in schema.items()
            }
        elif isinstance(schema, list):
            return [
                SchemaResolver.resolve_refs(item, definitions)
                for item in schema
            ]
        return schema
```

#### 2. 상세한 로깅

```python
import logging

logger = logging.getLogger(__name__)

logger.info(f"🔌 MCP 서버: {server_name}")
logger.info(f"  📋 도구 발견: {len(tools)} 개")

for tool in tools:
    logger.info(f"  ✅ {tool.name}")
    logger.debug(f"    원본 스키마: {tool.inputSchema}")
    logger.debug(f"    해소 후: {resolved}")
```

#### 3. 에러 핸들링

```python
try:
    definitions = tool.inputSchema.get("$defs", {})
    resolved = self._resolve_schema_refs(
        tool.inputSchema.get("properties", {}),
        definitions
    )
except KeyError as e:
    logger.error(f"❌ $ref 해소 실패: {e}")
    # Fallback: 원본 스키마 사용
    resolved = tool.inputSchema.get("properties", {})
except Exception as e:
    logger.error(f"❌ 예상치 못한 오류: {e}")
    # Fallback 또는 스킵
```

#### 4. 단위 테스트

```python
import unittest

class TestSchemaResolver(unittest.TestCase):
    def test_resolve_simple_ref(self):
        schema = {"$ref": "#/$defs/Person"}
        definitions = {
            "Person": {
                "properties": {"name": {"type": "string"}}
            }
        }
        result = resolve_refs(schema, definitions)
        self.assertEqual(result, {"properties": {"name": {"type": "string"}}})
    
    def test_resolve_nested_ref(self):
        schema = {"$ref": "#/$defs/Company"}
        definitions = {
            "Company": {
                "properties": {
                    "ceo": {"$ref": "#/$defs/Person"}
                }
            },
            "Person": {
                "properties": {"name": {"type": "string"}}
            }
        }
        result = resolve_refs(schema, definitions)
        self.assertIn("ceo", result["properties"])
        self.assertIn("name", result["properties"]["ceo"]["properties"])

if __name__ == '__main__':
    unittest.main()
```

---

## 참고 자료

- [JSON Schema Specification - $ref](https://json-schema.org/understanding-json-schema/structuring.html#ref)
- [JSON Schema - $defs](https://json-schema.org/understanding-json-schema/structuring.html#defs)
- [Pydantic JSON Schema](https://docs.pydantic.dev/latest/concepts/json_schema/)
- [FastMCP Documentation](https://github.com/jlowin/fastmcp)
- [Model Context Protocol Spec](https://spec.modelcontextprotocol.io/)

---

**작성일**: 2025-12-28  
**최종 수정**: 2025-12-28  
**카테고리**: MCP, LLM Integration  
**태그**: #MCP #Pydantic #BaseModel #JSONSchema #Ref #FastMCP #defs #properties