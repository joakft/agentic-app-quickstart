# AI Agent Exercises

This repository contains my **solutions to the exercises** from the course _Mastering AI Agents and MCP: Build Enterprise Agentic Apps_.

It was originally **forked** from the instructor's repository: [teacher-repo-link](https://github.com/teacher-repo-link).  
👉 If you're looking for instructions on how to run or configure the environment, **refer to the original repository**.

## 📁 My Projects

My own weekly agentic applications are developed in the folder:
```bash
./agentic_app_quickstart/[WEEK]/solution/
```

## 🛠️ Environment

The development environment is pre-configured with:

- Python 3.13  
- `uv` – modern, fast Python package manager  
- `openai-agents` – core library for agentic applications  
- `marimo` – interactive notebook interface  
- `FastAPI` – API framework  
- Tooling: `ruff`, `pytest`  

## 🔑 API Keys

You'll need to set environment variables in a `.env` file:

```bash
# OpenAI API Configuration
OPENAI_API_ENDPOINT="api_url"
OPENAI_API_KEY="your_openai_api_key_here"
```

⚠️ `.env` is already in `.gitignore`. Don’t remove it or share credentials publicly.

## 📦 Package Management

This repo uses `uv` for dependency management:

```bash
# Install uv if needed
pip install -U pip uv

# Sync dependencies
uv sync

# Add new packages
uv add package-name

# (Optional) Create a virtual environment
uv venv

# Run with uv
uv run python examples/code/01_...
```

---

_This fork exists to document my personal learning and exercise solutions. The course content and original structure belong to the instructor._
