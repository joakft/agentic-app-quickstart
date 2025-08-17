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
    instructions="You handle the conversation. Data Loader handles you the data. " \
    "The main goal of the person talking with you is analyze some CSV data" \
    "so if you are asked about data, it is the CSV data that indeed you possess.", 
    handoffs = [data_loader, analyst],
    model=get_model(),
)