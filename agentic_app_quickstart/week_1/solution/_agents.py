import asyncio
from tools import *
from agents import Agent, Runner, set_tracing_disabled, function_tool, handoff, SQLiteSession
from agentic_app_quickstart.examples.helpers import get_model
import os
import gradio as gr

data_loader = Agent(
    name="Data Loader",
    instructions="You have access to all the data in CSV and can answer about it",
    model=get_model(), 
    tools=[get_csv_data]
)

analyst = Agent(
    name="Analyst",
    instructions = (
    "You have two tools:\n"
    "1) run_pandas_code(code: str) — operate on df_weather, df_sales, df_employee; assign output to `result`.\n"
    "2) run_matplotlib_code(code: str) — build plots with fig/ax; DO NOT call plt.show(); returns a base64 PNG.\n\n"
    "CRITICAL: If the user requests a chart/plot/visualization, you MUST call run_matplotlib_code. "
    "Do not only describe the plot. Default behavior is to execute tools and return their outputs."),
    tools=[run_pandas_code, run_matplotlib_code],
    handoffs = [data_loader],
    model=get_model(),
)


customer_facing = Agent(
    name="Customer Facing",
    instructions=(
        "You are the primary conversational interface and must speak in the tone of a consultant: "
        "professional, clear, and helpful.\n\n"
        "Your role is to guide the user through analyzing CSV data. You do not directly manipulate data, "
        "but you have access to specialized agents via handoff:\n"
        "- Data Loader: provides raw CSV data.\n"
        "- Analyst: executes pandas and matplotlib code on the data.\n\n"
        "CRITICAL:\n"
        "- If the user requests a chart, visualization, or plot, you must hand off to the Analyst. "
        "Do not attempt to describe plots yourself.\n"
        "- You can only access data or analysis results if you consult the appropriate agent. "
        "Always delegate tasks instead of fabricating information.\n\n"
        "Your main goal: manage the conversation smoothly, ensure the right agent is consulted, "
        "and present the results back to the user in a clear, consultant-like manner."
    ),
    handoffs=[data_loader, analyst],
    model=get_model(),
)
