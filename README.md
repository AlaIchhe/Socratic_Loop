# Socratic Loop —— 苏格拉底式学习循环

基于 LangGraph + Chainlit 的多智能体苏格拉底式学习引导系统。三个 LLM 智能体（批判者 / 精确化者 / 裁判）通过人在循环的交互，帮助用户深化对任意论题的理解。

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env  # 填入 API Key
python -m chainlit run app.py
# 浏览器打开 http://localhost:8000
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_MODEL` | `gpt-4o` | LLM 模型名称 |
| `LLM_BASE_URL` |（空） | LLM API 端点（OpenAI 兼容） |
| `LLM_API_KEY` |（空） | LLM API Key（优先于 OPENAI_API_KEY） |
| `OPENAI_API_KEY` |（空） | OpenAI API Key（回退） |
| `LLM_FORCE_JSON_MODE` | `false` | 强制 Referee 使用 JSON-mode（DeepSeek 等设为 true） |
| `PORT` | `8000` | Chainlit 服务端口 |

## 运行测试

```bash
pytest tests/ -v
```

## 架构

```
core/          # state + schemas + prompts + agents + graph
config/        # settings + model（配置 + LLM 工厂）
tests/         # mock-based 单元测试
app.py         # Chainlit 入口
```
