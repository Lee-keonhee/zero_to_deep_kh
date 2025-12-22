---
layout: post
title: MCP(Model Context Protocol) LLM 연결하기
summary: MCP(Model Context Protocol) LLM 연결하기
author: keonhee
date: 2025-12-16 10:00:00 +0900
category: MCP
keywords: MCP
permalink: /blog/MCP_4/
usemathjax: true
thumbnail: /assets/img/posts/overview_of_NLP_1.png
---
# 05. 로컬 LLM과 연결하기 (Hugging Face)

## 목표

여러분이 만든 MCP 서버를 로컬 LLM(Hugging Face)과 연결해서, LLM이 실제로 여러분의 도구를 사용할 수 있게 만듭니다.

---

## 사전 준비: 필요한 패키지 설치

### 1. Hugging Face 라이브러리 설치

```bash
# 필수 패키지
pip install transformers torch accelerate bitsandbytes

# 선택사항: GPU 가속 (CUDA 있는 경우)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**각 패키지의 역할:**

- `transformers`: Hugging Face 모델 로드
- `torch`: PyTorch 백엔드
- `accelerate`: 모델 로딩 최적화
- `bitsandbytes`: 양자화 (메모리 절약)

### 2. GPU vs CPU 확인

PyCharm 터미널에서:

```python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**결과:**

- `CUDA available: True` → GPU 사용 가능! 🚀
- `CUDA available: False` → CPU로 작동 (느리지만 가능)

### 3. 추천 모델

RAG 작업을 해보셨으니 이미 알고 계실 수도 있는 모델들:

**가벼운 모델 (CPU도 가능):**

- `microsoft/Phi-3-mini-4k-instruct` (3.8B)
- `google/gemma-2b-it` (2B)

**중간 모델 (GPU 권장):**

- `mistralai/Mistral-7B-Instruct-v0.2` (7B)
- `meta-llama/Llama-3.2-3B-Instruct` (3B)

**큰 모델 (GPU 필수, 양자화 권장):**

- `meta-llama/Meta-Llama-3-8B-Instruct` (8B)

---

## MCP 클라이언트 만들기

### 프로젝트 구조

```
mcp-learning/
├── servers/
│   ├── calculator_server.py
│   └── memo_server.py
└── client/
    ├── __init__.py
    └── hf_client.py  ← 새로 만들 파일
```

---

## 단계 1: Hugging Face MCP 클라이언트 만들기

`client/hf_client.py` 파일 생성:

````python
"""  
Hugging Face Transformers를 사용하는 MCP 클라이언트  
로컬 LLM이 MCP 도구를 사용할 수 있게 합니다  
"""  
  
import asyncio  
import json  
import torch  
from typing import Dict, List, Any  
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig  
from mcp.client.session import ClientSession  
from mcp.client.stdio import StdioServerParameters, stdio_client  
from dotenv import load_dotenv  
import os  
  
load_dotenv()  
token = os.environ["HF_TOKEN"]  
  
  
class HuggingFaceMCPClient:  
    """Hugging Face + MCP 통합 클라이언트"""  
  
    def __init__(self,  
                 # model_name: str = "Qwen/Qwen2.5-7B-Instruct"  
                 model_name: str = "Qwen/Qwen3-4B-Instruct-2507",  
                 use_4bit: bool = True,  
                 device: str = "auto"  
                 ):  
        """  
        :param model_name: 사용할 hugging face 모델  
        :param use_4bit: 4bit 양자화 사용 여부 (메모리 절약)  
        :param device: 'auto','cpu','cuda'        """        self.model_name = model_name  
        self.device = device  
  
        self.conversation_history = []  
        self.mcp_sessions = {}  
        self.mcp_contexts = {}  # context manager 저장  
        self.available_tools = []  
  
        print(f"🤖 모델 로딩 중: {model_name}")  
        print(f"   디바이스: {device}")  
        print(f"   양자화: {'4bit' if use_4bit else 'None'}")  
  
        # 양자화 설정  
        if use_4bit and torch.cuda.is_available():  
            quantization_config = BitsAndBytesConfig(  
                load_in_4bit=True,  
                bnb_4bit_compute_dtype=torch.float16,  
                bnb_4bit_use_double_quant=True,  
                bnb_4bit_quant_type="nf4"  
            )  
        else:  
            quantization_config = None  
  
        self.tokenizer = AutoTokenizer.from_pretrained(  
            model_name,  
            # token=token,  
            trust_remote_code=True  
        )  
  
        self.model = AutoModelForCausalLM.from_pretrained(  
            model_name,  
            quantization_config=quantization_config,  
            device_map=device,  
            # token=token,  
            trust_remote_code=True,  
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  
        )  
  
        print("모델 로딩 완료")  
  
    async def connect_mcp_server(self, server_name:str, server_path:str):  
        """MCP 서버에 연결"""  
        print(f"{server_name} 서버에 연결 중...")  
  
        server_params = StdioServerParameters(  
            command="python",  
            args=[server_path],  
        )  
  
        # Context manager를 저장  
        stdio_context = stdio_client(server_params)  
        read, write = await stdio_context.__aenter__()  
  
        session_context = ClientSession(read, write)  
        session = await session_context.__aenter__()  
  
        await session.initialize()  
  
        # 세션과 context 모두 저장  
        self.mcp_sessions[server_name] = session  
        self.mcp_contexts[server_name] = (stdio_context, session_context)  
  
        tools_response = await session.list_tools()  
        for tool in tools_response.tools:  
            self.available_tools.append({  
                "server": server_name,  
                "name": tool.name,  
                "description": tool.description,  
                "input_schema": tool.inputSchema,  
            })  
  
        print(f"✅ {server_name} 연결 완료! ({len(tools_response.tools)}개 연결 완료)")  
  
  
    def get_tools_description(self) -> str:  
        """도구 설명을 LLM에게 전달할 형식으로 변환"""  
        if not self.available_tools:  
            return "사용 가능한 도구가 없습니다."  
  
        tools_desc = "## Available tools:\n\n"  
  
        for tool in self.available_tools:  
            tools_desc += f"### {tool['name']} ({tool['server']})\n"  
            tools_desc += f"Description: {tool['description']}\n"  
            tools_desc += f"Input Schema: {json.dumps(tool['input_schema'], indent=2, ensure_ascii=False)}\n\n"  
            tools_desc += """  
            ## How to Use Tools  
            To use a tool, respond with a JSON code block like this:            If you decide to use a tool, DO NOT include any explanation text.            Respond with ONLY a JSON code block.            ```json            {                "tool_call": {                    "server": "server_name",                    "tool": "tool_name",                    "arguments": {                        "arg1": "value1",                        "arg2": "value2"                    }                }            }            ```  
            If you don't need to use a tool, just respond normally in Korean.            """  
        return tools_desc  
  
    async def call_mcp_tool(self, server_name:str, tool_name:str, arguments:dict) -> str:  
        """MCP 도구 실행"""  
        if server_name not in self.mcp_sessions:  
            return f"오류: {server_name} 서버를 찾을 수 없습니다."  
  
        try:  
            session = self.mcp_sessions[server_name]  
            result = await session.call_tool(tool_name, arguments)  
            return result.content[0].text  
  
        except Exception as e:  
            return f"❌ 도구 실행 오류 :{str(e)}"  
  
    def parse_tool_call(self, response:str) -> Dict[str, Any] | None:  
        """LLM 응답에서 도구 호출 파싱"""  
        if "```json" not in response:  
            return None  
        try:  
            start = response.find("```json") + len("```json")  
            end = response.find("```", start)  
            json_str = response[start:end].strip()  
  
            data = json.loads(json_str)  
  
            if "tool_call" in data:  
                return data["tool_call"]  
  
            return None  
        except Exception as e:  
            print(f"도구 호출 파싱 실패 : {e}")  
            return None  
  
    def generate_response(self, messages:List[Dict[str,str]], max_new_tokens:int=512,) ->str:  
        """LLM 응답 생성"""  
  
        # 메시지를 프롬프트로 변환  
        prompt = self._format_messages(messages)  
  
        # Tokenizing  
        inputs = self.tokenizer(prompt,  
                                return_tensors="pt",  
                                truncation=True,  
                                max_length=2048)  
  
        inputs = {k:v.to(self.model.device) for k, v in inputs.items()}  
  
        with torch.no_grad():  
            outputs = self.model.generate(  
                **inputs,  
                max_new_tokens=max_new_tokens,  
                do_sample=True,  
                temperature=0.7,  
                top_p=0.9,  
                pad_token_id=self.tokenizer.eos_token_id  
            )  
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)  
  
        response = full_response[len(prompt):].strip()  
  
        return response  
  
  
    def _format_messages(self, messages:List[Dict[str,str]]) -> str:  
        """메시지를 프롬프트 형식으로 변환"""  
        # 모델에 따라 다른 형식 사용  
        if "Phi" in self.model_name or "phi" in self.model_name:  
            # Phi 형식  
            prompt = ""  
            for msg in messages:  
                if msg["role"] == "system":  
                    prompt += f"<|system|>\n{msg['content']}<|end|>\n"  
                elif msg["role"] == "user":  
                    prompt += f"<|user|>\n{msg['content']}<|end|>\n"  
                elif msg["role"] == "assistant":  
                    prompt += f"<|assistant|>\n{msg['content']}<|end|>\n"  
            prompt += "<|assistant|>\n"  
  
        elif "Llama" in self.model_name or "llama" in self.model_name:  
            # Llama 형식  
            prompt = ""  
            for msg in messages:  
                if msg["role"] == "system":  
                    prompt += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{msg['content']}<|eot_id|>\n"  
                elif msg["role"] == "user":  
                    prompt += f"<|start_header_id|>user<|end_header_id|>\n{msg['content']}<|eot_id|>\n"  
                elif msg["role"] == "assistant":  
                    prompt += f"<|start_header_id|>assistant<|end_header_id|>\n{msg['content']}<|eot_id|>\n"  
            prompt += "<|start_header_id|>assistant<|end_header_id|>\n"  
  
        else:  
            # 기본 형식  
            prompt = ""  
            for msg in messages:  
                prompt += f"{msg['role'].capitalize()}: {msg['content']}\n\n"  
            prompt += "Assistant: "  
  
        return prompt  
  
    async def chat(self, user_message: str) -> str:  
        """사용자와 대화"""  
        print(f"\n💬 사용자: {user_message}")  
  
        # 대화 히스토리에 추가  
        self.conversation_history.append({  
            "role": "user",  
            "content": user_message  
        })  
  
        # 시스템 프롬프트 (도구 설명 포함)  
        system_prompt = f"""You are a helpful AI assistant with access to tools.  
  
        {self.get_tools_description()}  
  
        Analyze the user's request and:  
        1. If a tool is needed, use the JSON format above to call it.  
        2. If no tool is needed, respond naturally in Korean.  
        3. Always communicate in Korean with the user. """  
  
        # 메시지 준비  
        messages = [  
            {"role": "system", "content": system_prompt},  
            *self.conversation_history[-5:]  
        ]  
  
        # LLM 응답 생성  
        print("🤔 AI가 생각하는 중...")  
        llm_response = self.generate_response(messages)  
  
        # 도구 호출 확인  
        tool_call = self.parse_tool_call(llm_response)  
  
        if tool_call:  
            print(f"🔧 도구 사용: {tool_call['tool']}")  
  
            # 도구 실행  
            tool_result = await self.call_mcp_tool(  
                tool_call['server'],  
                tool_call['tool'],  
                tool_call['arguments']  
            )  
  
            print(f"📊 도구 결과:\n{tool_result}")  
  
            # 도구 결과를 히스토리에 추가  
            self.conversation_history.append({  
                "role": "assistant",  
                "content": f"[Tool used: {tool_call['tool']}]"  
            })  
  
            # 도구 결과를 포함한 최종 답변 생성  
            final_messages = [  
                {"role": "system",  
                 "content": "Based on the tool execution result, provide a natural response to the user in Korean."},  
                {"role": "user", "content": user_message},  
                {"role": "assistant", "content": f"Tool result: {tool_result}\n\nNow I'll respond:"}  
            ]  
  
            final_answer = self.generate_response(final_messages, max_new_tokens=256)  
            print(f"\n🤖 AI: {final_answer}")  
  
            self.conversation_history.append({  
                "role": "assistant",  
                "content": final_answer  
            })  
  
            return final_answer  
  
        else:  
            # 도구 없이 일반 대화  
            print(f"\n🤖 AI: {llm_response}")  
  
            self.conversation_history.append({  
                "role": "assistant",  
                "content": llm_response  
            })  
  
            return llm_response  
  
    async def close(self):  
        """모든 MCP 세션 종료"""  
        for server_name, (stdio_ctx, session_ctx) in self.mcp_contexts.items():  
            await session_ctx.__aexit__(None, None, None)  
            await stdio_ctx.__aexit__(None, None, None)  
  
        self.mcp_sessions.clear()  
        self.mcp_contexts.clear()  
  
  
async def main():  
    """메인 함수"""  
    print("=" * 70)  
    print("🚀 Hugging Face MCP 클라이언트 시작")  
    print("=" * 70)  
  
    # 모델 선택 (여기서 변경 가능)  
    # MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"    # MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"  # 다른 모델 사용 시  
    MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"  
    # 클라이언트 생성  
    client = HuggingFaceMCPClient(  
        model_name=MODEL_NAME,  
        use_4bit=True,  # GPU 메모리 절약  
        device="auto"  # 자동으로 GPU/CPU 선택  
    )  
  
    # MCP 서버들 연결  
    await client.connect_mcp_server(  
        "calculator",  
        "servers/calculator_server.py"  
    )  
    # await client.connect_mcp_server(  
    #     "memo",    #     "servers/memo_server.py"    # )  
    print("\n사용 가능한 도구:")  
    for tool in client.available_tools:  
        print(f"  - {tool['name']} ({tool['server']}): {tool['description']}")  
  
    print("\n" + "=" * 70)  
    print("대화를 시작합니다! ('exit'를 입력하면 종료)")  
    print("=" * 70)  
  
    # 대화 루프  
    try:  
        while True:  
            user_input = input("\n당신: ").strip()  
  
            if user_input.lower() in ['exit', 'quit', '종료']:  
                print("\n👋 대화를 종료합니다.")  
                break  
  
            if not user_input:  
                continue  
  
            await client.chat(user_input)  
  
    finally:  
        await client.close()  
  
  
if __name__ == "__main__":  
    asyncio.run(main())

````

---

## 단계 2: 실행해보기

### 터미널에서 실행

```bash
python client/hf_client.py
```

### 첫 실행 시 주의사항

**모델 다운로드:**

- 처음 실행하면 모델을 다운로드합니다 (수 GB)
- 인터넷 연결 필요
- 다운로드 후에는 오프라인 사용 가능

**예상 출력:**

```
======================================================================
🚀 Hugging Face MCP 클라이언트 시작
======================================================================
🤖 모델 로딩 중: microsoft/Phi-3-mini-4k-instruct
   디바이스: auto
   양자화: 4bit

Downloading model... (처음만 나타남)
✅ 모델 로딩 완료!

🔌 calculator 서버에 연결 중...
✅ calculator 연결 완료! (2개 도구)
🔌 memo 서버에 연결 중...
✅ memo 연결 완료! (4개 도구)

사용 가능한 도구:
  - add (calculator): 두 숫자를 더합니다
  - subtract (calculator): 두 숫자를 뺍니다
  - save_memo (memo): 새로운 메모를 저장합니다
  - get_memo (memo): 특정 ID의 메모를 가져옵니다
  - list_memos (memo): 저장된 모든 메모의 목록을 반환합니다
  - search_memos (memo): 키워드로 메모를 검색합니다

======================================================================
대화를 시작합니다! ('exit'를 입력하면 종료)
======================================================================

당신: 
```

---

## 단계 3: 실제 대화 예시

### 예시 1: 계산

```
당신: 123과 456을 더해줘

💬 사용자: 123과 456을 더해줘
🤔 AI가 생각하는 중...
🔧 도구 사용: add
📊 도구 결과:
123 + 456 = 579

🤖 AI: 123과 456을 더한 결과는 579입니다.
```

### 예시 2: 메모 저장

```
당신: "MCP 학습 완료"라는 제목으로 메모를 저장해줘. 내용은 "Hugging Face로 MCP 서버와 연결했다"이고 ID는 memo_hf_001로

💬 사용자: "MCP 학습 완료"라는...
🤔 AI가 생각하는 중...
🔧 도구 사용: save_memo
📊 도구 결과:
✅ 메모 저장 완료!
ID: memo_hf_001
제목: MCP 학습 완료

🤖 AI: 메모가 성공적으로 저장되었습니다!
```

---

## 성능 최적화

### 1. GPU 메모리 부족 시

더 작은 모델 사용:

```python
MODEL_NAME = "google/gemma-2b-it"  # 2B 모델
```

또는 CPU 사용:

```python
client = HuggingFaceMCPClient(
    model_name=MODEL_NAME,
    use_4bit=False,
    device="cpu"
)
```

### 2. 응답 속도 개선

생성 토큰 수 줄이기:

```python
final_answer = self.generate_response(final_messages, max_new_tokens=128)
```

### 3. 컨텍스트 관리

대화 히스토리 제한:

```python
# hf_client.py의 chat 메서드에서
messages = [
    {"role": "system", "content": system_prompt},
    *self.conversation_history[-3:]  # 최근 3개만
]
```

---

## 디버깅: 문제 해결

### 문제 1: "CUDA out of memory"

**해결책 1: 4bit 양자화 활성화**

```python
client = HuggingFaceMCPClient(use_4bit=True)
```

**해결책 2: 더 작은 모델**

```python
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"  # 3.8B
# 대신
MODEL_NAME = "google/gemma-2b-it"  # 2B
```

**해결책 3: CPU 사용**

```python
client = HuggingFaceMCPClient(device="cpu", use_4bit=False)
```

### 문제 2: "Model not found"

**해결:**

```python
# Hugging Face에 로그인 (private 모델 접근 시)
from huggingface_hub import login
login("your_token_here")
```

토큰 생성: https://huggingface.co/settings/tokens

### 문제 3: LLM이 도구를 사용하지 않음

**원인:**

- 모델이 instruction following을 잘 못함
- 프롬프트가 불명확

**해결책 1: 더 명확하게 요청**

```
"add 도구를 사용해서 5와 3을 더해줘"
```

**해결책 2: 더 나은 모델 사용**

```python
# Instruction-tuned 모델 사용
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
```

**해결책 3: 프롬프트 엔지니어링**

```python
# _format_messages에서 더 명확한 예시 추가
```

### 문제 4: 느린 응답 속도

**해결:**

```python
# 생성 파라미터 조정
outputs = self.model.generate(
    **inputs,
    max_new_tokens=256,  # 줄이기
    do_sample=False,      # sampling 끄기 (더 빠름)
    num_beams=1           # beam search 끄기
)
```

---

## 모델 비교

여러분의 환경에 맞는 모델 선택하세요:

|모델|크기|GPU 메모리|속도|성능|권장 사용|
|---|---|---|---|---|---|
|google/gemma-2b-it|2B|~4GB|빠름|보통|CPU 또는 저사양 GPU|
|microsoft/Phi-3-mini-4k|3.8B|~6GB|보통|좋음|중급 GPU|
|meta-llama/Llama-3.2-3B|3B|~5GB|보통|좋음|중급 GPU|
|mistralai/Mistral-7B|7B|~10GB|느림|매우좋음|고급 GPU|

**4bit 양자화 사용 시 메모리는 약 1/4로 감소**

---

## 추가 기능: 다양한 모델 지원

`client/hf_client.py`에 모델 선택 기능 추가:

```python
def select_model():
    """사용자가 모델을 선택하게 함"""
    models = {
        "1": ("microsoft/Phi-3-mini-4k-instruct", "Phi-3 Mini (3.8B) - 균형잡힌 성능"),
        "2": ("google/gemma-2b-it", "Gemma 2B - 가볍고 빠름"),
        "3": ("meta-llama/Llama-3.2-3B-Instruct", "Llama 3.2 (3B) - 좋은 성능"),
        "4": ("mistralai/Mistral-7B-Instruct-v0.2", "Mistral 7B - 최고 성능 (GPU 필요)")
    }
    
    print("\n사용할 모델을 선택하세요:")
    for key, (name, desc) in models.items():
        print(f"{key}. {desc}")
    
    choice = input("\n선택 (1-4): ").strip()
    return models.get(choice, models["1"])[0]

# main() 함수에서
MODEL_NAME = select_model()
```

---

## Hugging Face vs Ollama 비교

### Hugging Face 장점 ✅

- **완전한 제어**: 모든 파라미터 조정 가능
- **다양한 모델**: 수천 개 선택 가능
- **커스터마이징**: Fine-tuning, 양자화 등
- **RAG 친화적**: 이미 익숙한 생태계

### Hugging Face 단점 ❌

- **복잡한 설정**: GPU, CUDA, 의존성 관리
- **메모리 관리**: 직접 해야 함
- **디버깅**: 더 많은 노력 필요

### Ollama 장점 ✅

- **간단한 설정**: 한 줄 설치
- **자동 최적화**: 메모리 관리 자동
- **사용 편의성**: 초보자 친화적

### Ollama 단점 ❌

- **제한된 선택**: 지원 모델 한정
- **커스터마이징 제한**: 세밀한 조정 어려움

**여러분의 선택:**

- RAG 경험 있고 세밀한 제어 원함 → Hugging Face ✅
- 빠른 프로토타이핑 원함 → Ollama

---

## 이해도 확인

다음 질문에 답할 수 있나요?

1. **4bit 양자화란 무엇이고 왜 사용하나요?**
    
    - 힌트: 메모리 절약
2. **프롬프트 형식이 모델마다 다른 이유는?**
    
    - 힌트: 각 모델이 학습된 방식
3. **max_new_tokens 파라미터의 역할은?**
    
    - 힌트: 생성할 토큰 수
4. **CPU vs GPU 사용 시 차이점은?**
    
    - 힌트: 속도

---

## 체크포인트

다음 항목을 모두 완료했나요?

- [ ] transformers, torch 등 패키지 설치
- [ ] GPU 사용 가능 여부 확인
- [ ] hf_client.py 작성
- [ ] 모델 다운로드 및 로딩 성공
- [ ] MCP 서버와 연결
- [ ] LLM과 대화하며 도구 사용 확인

---

## 축하합니다! 🎉

여러분은 이제:

- ✅ Hugging Face 모델을 로컬에서 실행할 수 있습니다
- ✅ MCP 서버를 커스텀 클라이언트에 연결할 수 있습니다
- ✅ LLM이 도구를 사용하게 만들 수 있습니다
- ✅ 완전히 로컬에서 작동하는 AI 시스템을 구축했습니다
- ✅ RAG 경험을 MCP로 확장했습니다

---

## 다음 단계


**다음: [05. MCP(Model Context Protocol)-API-통합](https://lee-keonhee.github.io/zero_to_deep_kh/blog/MCP_5/#/)**

여러분이 LangGraph로 만들었던 RAG를 MCP 서버로 전환하고, Hugging Face LLM과 연결해봅시다!

