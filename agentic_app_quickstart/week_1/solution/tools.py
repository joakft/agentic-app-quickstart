
import pathlib
from agents import function_tool
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import io, base64, tempfile, uuid, traceback

def read_csv(file):
    current_dir = str(pathlib.Path(__file__).resolve().parent)
    with open(current_dir+file, 'r') as f:
        contents = f.readlines()
    return contents

DATA_DIR = pathlib.Path(__file__).parent / "data"

df_sales = pd.read_csv(DATA_DIR / "sample_sales.csv")
df_employee = pd.read_csv(DATA_DIR / "employee_data.csv")
df_weather = pd.read_csv(DATA_DIR / "weather_data.csv")

@function_tool
def get_csv_data():
    sample_sales_data = read_csv('/data/sample_sales.csv')
    employee_data = read_csv('/data/employee_data.csv')
    weather_data = read_csv('/data/weather_data.csv')
    all_data = {
        'sample_sales_data' : sample_sales_data,
        'employee_data' : employee_data,
        'weather_data': weather_data,
    }
    return str(all_data)


@function_tool
def run_pandas_code(code: str) -> str:
    """
    Executes arbitrary pandas code on the preloaded DataFrame 'df'.
    The code must assign its final result to a variable called 'result'.

    for example sending
    code = "result = df.groupby("age")["salary"].mean()"
    then
    run_pandas_code(code)
    """

    local_vars = {"df_sales": df_sales.copy(),
                  "df_employee": df_employee.copy(),
                  "df_weather": df_weather.copy(),
                  "pd": pd}
    try:
        exec(code, {}, local_vars)
        result = local_vars.get("result", None)
        if isinstance(result, pd.DataFrame):
            return result.to_string()
        elif isinstance(result, pd.Series):
            return result.to_string()
        else:
            return str(result)
    except Exception as e:
        return f"Error: {e}"
    
LAST_FILE = None
LAST_SEQ = 0

@function_tool
def run_matplotlib_code(code: str) -> dict | str:
    """
    Execute matplotlib code using df_* with fig/ax available.
    Returns:
      - On success: {"path": "<server-filepath>", "mime_type":"image/png", "alt_text":"..."}
      - On error:   "Error: <message>"
    """
    global LAST_FILE, LAST_SEQ
    plt.close("all")  # clear state before execution
    try:
        # Prepare the execution environment
        local_vars = {
            "df_sales": df_sales.copy(),
            "df_employee": df_employee.copy(),
            "df_weather": df_weather.copy(),
            "pd": pd,
            "plt": plt
        }
        # Provide fig/ax if the user wants them
        fig, ax = plt.subplots()
        local_vars.update({"fig": fig, "ax": ax})

        print("[DEBUG] Executing matplotlib code:\n", code)
        exec(code, {}, local_vars)

        # After exec, try to grab the last active figure
        fig_to_save = plt.gcf()
        if fig_to_save is None:
            return "Error: No figure was generated."

        # Save to a temp file
        tmpdir = pathlib.Path(tempfile.gettempdir())
        fpath = tmpdir / f"plot-{uuid.uuid4().hex}.png"
        fig_to_save.savefig(fpath, format="png", bbox_inches="tight", dpi=144)

        block = {
            "path": str(fpath),
            "mime_type": "image/png",
            "alt_text": "Generated plot"
        }
        LAST_FILE = block
        LAST_SEQ += 1
        print("[DEBUG] Plot file saved:", fpath)
        return block

    except Exception as e:
        print("[ERROR] run_matplotlib_code failed:\n", traceback.format_exc())
        LAST_FILE = None
        return f"Error: {e}"
    finally:
        plt.close("all")
