# Socratic Loop —— 苏格拉底式学习循环

基于 LangGraph + Chainlit 的多智能体苏格拉底式学习引导系统。三个 LLM 智能体（批判者 / 精确化者 / 裁判）通过人在循环的交互，帮助用户深化对任意论题的理解。

## 快速开始

```bash
uv sync --group dev
cp .env.example .env  # 填入 API Key（也可在浏览器 Settings 面板配置）
uv run chainlit run app.py
# 浏览器打开 http://localhost:8000
```

启动后在浏览器 Settings 面板选择 Provider、填入 API Key，点击「🔄 获取模型列表」即可刷新当前可用模型。
