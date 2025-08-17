from agents import Runner, set_tracing_disabled, SQLiteSession
from _agents import analyst
import gradio as gr
import tools 
import base64
_last_seen_seq = -1  # module-level


set_tracing_disabled(True)

session = SQLiteSession("user_123", "conversations.db")



async def ask_agent(user_message: str):
    res = await Runner.run(analyst, user_message, session=session)  # ensure this is the agent with tools
    return getattr(res, "final_output", res)

async def respond(message, history):
    seq_before = tools.LAST_SEQ
    result = await ask_agent(message)
    print("[DEBUG] agent returned:", type(result), repr(result)[:200])

    # If the plotting tool ran this turn, show the image regardless of LLM narration
    if tools.LAST_FILE is not None and tools.LAST_SEQ != seq_before:
        return {"role": "assistant", "content": tools.LAST_FILE}  # content is a FILE DICT, not a list

    # If the tool already returned a proper file dict as final_output
    if isinstance(result, dict) and "path" in result:
        return {"role": "assistant", "content": result}

    # Otherwise just show the text
    return str(result)

gr.ChatInterface(fn=respond, title="Data Analyst", type="messages").launch(server_name="0.0.0.0", server_port=7860)