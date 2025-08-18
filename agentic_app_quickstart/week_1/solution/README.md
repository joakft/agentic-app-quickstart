# Week 1 — Multi-Agent Data Analyst App

This folder contains my solution for **Week 1** of the _Mastering AI Agents and MCP_ course.

It implements a **multi-agent system** where agents collaborate through `handoff` to help a user analyze CSV data interactively.  
The app combines **chat**, **code execution** (`pandas` + `matplotlib`), and **persistent memory**.

---

## 🤖 Agents

### 🗂️ Data Loader
- Role: Provides direct access to the CSV files.
- Tools:  
  - `get_csv_data()` — returns raw CSV contents.  
- Responsibility: Responds when raw data access is required.

---

### 📊 Analyst
- Role: Executes code and performs analysis.  
- Tools:  
  - `run_pandas_code(code: str)` — manipulates `df_sales`, `df_employee`, `df_weather`.  
  - `run_matplotlib_code(code: str)` — generates plots (saved as PNGs).  
- **Critical behavior**:  
  - If the user asks for a chart or visualization, the Analyst **must** use `run_matplotlib_code`.  
  - It should not only describe visuals, but actually create them.  
- Handoffs: Can consult `Data Loader` for raw CSVs.

---

### 💼 Customer Facing
- Role: The **primary conversational interface**.  
- Tone: **Consultant-like** — professional, clear, and helpful.  
- Responsibilities:  
  - Guides the conversation and ensures the user’s goals are understood.  
  - Delegates tasks to the appropriate agent:
    - → `Data Loader` when raw CSVs are needed.  
    - → `Analyst` for data processing or plots.  
  - **Critical**:  
    - Must **handoff to Analyst** for any plot/visualization request.  
    - Must not fabricate results — always consult the right agent.  
- Returns the results back to the user in a polished and explanatory way.

---

## 🧠 Memory

Session state is persisted with:

```python
SQLiteSession(user_id="user_123", db_path="conversations.db")
```

This enables multi-turn memory so that analyses can build on prior conversation.

---

## 💬 Frontend

- The app runs through a **Gradio ChatInterface** (`main.py`).  
- User messages are routed first to `Customer Facing`, then delegated via `handoff`.  
- Responses can be:
  - Natural language,
  - DataFrame summaries,
  - or plot images (rendered automatically in the chat).

---

## ▶️ How to Run

```bash
uv run python main.py
```

Then open: [http://localhost:7860](http://localhost:7860)

---

## 📁 File Structure

```bash
week_01/
├── main.py                  # Entry point — Gradio + async routing
├── _agents/
│   └── analyst.py           # Agent definitions
├── tools.py                 # Tools: CSV, pandas, matplotlib
├── data/
│   ├── sample_sales.csv
│   ├── employee_data.csv
│   └── weather_data.csv
├── conversations.db         # SQLite persistent session memory
```

---

## 📦 Tech Stack

- Python 3.13  
- [uv](https://github.com/astral-sh/uv) — fast dependency manager  
- [Gradio](https://www.gradio.app/) — chat frontend  
- `pandas` & `matplotlib` — data + visualization tools  
- `openai-agents` — orchestration with tools and handoffs  

---

_This solution demonstrates how multiple agents can collaborate through delegation: a consultant-style interface (`Customer Facing`), a technical data expert (`Analyst`), and a supporting data provider (`Data Loader`). Together they provide a flexible and user-friendly data analysis experience._
