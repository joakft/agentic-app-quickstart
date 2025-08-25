from agents import Runner, set_tracing_disabled, SQLiteSession
from agents.result import RunResult

from _agents import analyst, customer_facing
import gradio as gr
import tools
import time

# -----------------------------
# PHOENIX + OpenTelemetry setup
# -----------------------------
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor

tracer_provider = register(
    project_name="agent_analyst",
    endpoint="https://app.phoenix.arize.com/s/joakft/v1/traces",
    auto_instrument=True,
)

try:
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
except Exception as e:
    print("[WARN] OpenAI already instrumented:", e)

# -----------------------------

trace_log = []
set_tracing_disabled(True)   # disable agent’s own tracing
session = SQLiteSession("user_123", "conversations.db")


# -----------------------------
# Core agent runner helpers
# -----------------------------
async def ask_agent(user_message: str):
    """Run the customer-facing agent and return its raw RunResult."""
    return await Runner.run(customer_facing, user_message, session=session)


async def respond(message, history):
    """Return only what should appear in the chat window (raw agent output)."""
    seq_before = tools.LAST_SEQ
    result = await ask_agent(message)
    print("[DEBUG] agent returned:", type(result), repr(result)[:200])

    # If plotting tool ran this turn
    if tools.LAST_FILE is not None and tools.LAST_SEQ != seq_before:
        return {"role": "assistant", "content": tools.LAST_FILE}, result

    # If tool directly returned a file dict
    if isinstance(result, dict) and "path" in result:
        return {"role": "assistant", "content": result}, result

    # If it's a RunResult → just show its final_output string
    if isinstance(result, RunResult):
        return {"role": "assistant", "content": str(result.final_output)}, result

    # Fallback
    return {"role": "assistant", "content": str(result)}, result


async def respond_with_trace(message, history):
    """Wrap respond() with latency logging + detailed trace log."""
    start = time.time()
    assistant_msg, raw_result = await respond(message, history)
    duration = time.time() - start

    updated_history = history + [
        {"role": "user", "content": message},
        assistant_msg,
    ]

    # Build trace log entry
    trace_entry = f"User: {message} → Agent replied in {duration:.2f}s"

    if isinstance(raw_result, RunResult):
        steps = []
        for item in getattr(raw_result, "input", []):
            # Agent/user/assistant messages
            if hasattr(item, "text"):
                steps.append(f"[Message] {getattr(item, 'text', '')[:80]}")

            # Tool calls
            elif hasattr(item, "args"):
                fn = getattr(item, "name", "unknown_tool")
                args = getattr(item, "args", {})
                out_preview = str(getattr(item, "output", ""))[:60]
                steps.append(f"[Tool] {fn}({args}) → {out_preview}")

            # Handoffs
            elif hasattr(item, "agent"):
                agent_name = getattr(getattr(item, "agent", None), "name", "unknown")
                steps.append(f"[Handoff] {agent_name}")

            # Fallback
            else:
                steps.append(str(item))

        if steps:
            trace_entry += "\n  → " + "\n  → ".join(steps[-8:])



    trace_log.append(trace_entry)
    return updated_history, "\n".join(trace_log)


# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks() as demo:
    with gr.Row():
        chat = gr.Chatbot(label="Data Analyst", type="messages")
        trace_console = gr.Textbox(label="Trace Log", lines=20, interactive=False)

    with gr.Row():
        msg = gr.Textbox(label="Your message")
        send = gr.Button("Send")

    send.click(
        fn=respond_with_trace,
        inputs=[msg, chat],
        outputs=[chat, trace_console],
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
