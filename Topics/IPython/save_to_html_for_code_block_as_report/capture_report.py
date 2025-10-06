# capture_results.py

import os
import psutil
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
import base64
from datetime import datetime
import contextlib
import sys
import inspect
import matplotlib
# matplotlib.use("Agg")
from IPython.display import display
import builtins
import ast
import textwrap


# --- Global defaults ---
OUTPUT_FILE = "report.html"
_cpu_times, _wall_times, _mem_used, _plot_labels = [], [], [], []


# --- Initialize report ---
def init_report(report_file="report.html", template_path="template.html",
                notebook_name="Untitled Notebook", description=""):
    """Initialize the report and insert notebook name, timestamp, and description."""
    global _cpu_times, _wall_times, _plot_labels, _mem_used, OUTPUT_FILE
    OUTPUT_FILE = report_file
    _cpu_times, _wall_times, _plot_labels, _mem_used = [], [], [], []

    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    summary_html = f"""
    <p><b>Notebook:</b> {notebook_name}</p>
    <p><b>Run at:</b> {run_time}</p>
    <p>{description}</p>
    <div id="summary-metrics"></div>
    """

    updated = template.replace('<p>General notes and overview of the analysis go here.</p>', summary_html)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(updated)


# --- Utilities ---
def _fig_to_base64(fig, width_px=1000):
    """Convert a matplotlib figure to a base64 HTML image."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=100)
    plt.close(fig)
    img_base64 = base64.b64encode(buf.getvalue()).decode()
    return f"<img src='data:image/png;base64,{img_base64}' width='{width_px}px'>"


def _inject(section_id, html, report_file=None, prepend=False):
    """Inject HTML into a div by ID in the report file."""
    target_file = report_file or OUTPUT_FILE
    with open(target_file, "r", encoding="utf-8") as f:
        content = f.read()
    marker = f'<div id="{section_id}">'
    if marker not in content:
        raise ValueError(f"Marker {marker} not found in {target_file}.")

    if prepend:
        updated = content.replace(marker, marker + html, 1)
    else:
        close_index = content.find("</div>", content.find(marker))
        if close_index == -1:
            raise ValueError(f"Closing div not found for {section_id}.")
        updated = content[:close_index] + html + content[close_index:]

    with open(target_file, "w", encoding="utf-8") as f:
        f.write(updated)


# --- Tee output context manager ---
@contextlib.contextmanager
def tee_output(stdout_buffer, stderr_buffer, display_live=False):
    class TeeStream:
        def __init__(self, original, buffer):
            self.original = original
            self.buffer = buffer
        def write(self, data):
            if display_live:
                self.original.write(data)
                self.original.flush()
            self.buffer.write(data)
        def flush(self):
            if display_live:
                self.original.flush()
            self.buffer.flush()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = TeeStream(sys.stdout, stdout_buffer)
    sys.stderr = TeeStream(sys.stderr, stderr_buffer)
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


# --- Run code and capture all outputs ---
def run_and_capture_all(key, code, report_file=None, title=None, display_live=False, ns=None):
    """
    Run code, capture stdout/stderr, exceptions, matplotlib figures, and displayed objects.
    Each plt.show() generates a figure in the HTML and is shown in the notebook.
    """

    stdout_buf, stderr_buf = StringIO(), StringIO()
    local_ns = {}

    if ns is None:
        caller_frame = inspect.currentframe().f_back
        exec_ns = caller_frame.f_globals
    else:
        exec_ns = ns

    start_wall = datetime.now()
    start_cpu = psutil.Process(os.getpid()).cpu_times().user

    html_content = "<pre>\n"
    figures_before = set(plt.get_fignums())

    # --- Patch plt.show to capture figures and display in notebook ---
    _original_show = plt.show
    def _patched_show(*args, **kwargs):
        figs = [plt.figure(num) for num in plt.get_fignums()]
        for fig in figs:
            html_content_piece = _fig_to_base64(fig)
            nonlocal html_content
            html_content += html_content_piece
            display(fig)  # notebook display
        _original_show(*args, **kwargs)

    plt.show = _patched_show

    # --- Patch display() to capture objects explicitly displayed ---
    displayed_objects = []
    _original_display = display
    def _patched_display(obj=None, *args, **kwargs):
        if obj is not None:
            displayed_objects.append(obj)
        _original_display(obj, *args, **kwargs)
    builtins.display = _patched_display

    # --- Transform last expression to capture its value ---
    code_ast = ast.parse(textwrap.dedent(code), mode="exec")
    if code_ast.body and isinstance(code_ast.body[-1], ast.Expr):
        last_expr = code_ast.body[-1]
        assign = ast.Assign(
            targets=[ast.Name(id="_last_expr_value", ctx=ast.Store(),
                              lineno=last_expr.lineno, col_offset=last_expr.col_offset)],
            value=last_expr.value,
            lineno=last_expr.lineno,
            col_offset=last_expr.col_offset
        )
        code_ast.body[-1] = assign
    compiled = compile(code_ast, filename="<string>", mode="exec")

    # --- Execute code with tee for stdout/stderr ---
    with tee_output(stdout_buf, stderr_buf, display_live=display_live):
        try:
            exec(compiled, exec_ns, local_ns)
        except Exception as e:
            stderr_buf.write(f"Exception: {repr(e)}\n")

    # Restore patched functions
    plt.show = _original_show
    builtins.display = _original_display

    # Capture any remaining figures (not shown)
    figures_after = plt.get_fignums()
    new_figs = [plt.figure(num) for num in figures_after if num not in figures_before]
    for fig in new_figs:
        html_content += _fig_to_base64(fig)
        plt.close(fig)

    end_wall = datetime.now()
    end_cpu = psutil.Process(os.getpid()).cpu_times().user
    wall_sec = (end_wall - start_wall).total_seconds()
    cpu_sec = end_cpu - start_cpu
    mem_mb = psutil.Process(os.getpid()).memory_info().rss / (1024**2)

    block_title = title or key

    if stdout_buf.getvalue():
        html_content += f"\nstdout:\n{stdout_buf.getvalue()}\n"
    if stderr_buf.getvalue():
        html_content += f"\nstderr:\n{stderr_buf.getvalue()}\n"
    html_content += "</pre>\n"

    # --- Include displayed objects and last expression ---
    for obj in displayed_objects:
        if isinstance(obj, plt.Figure):
            html_content += _fig_to_base64(obj)
        elif "pandas" in str(type(obj)):
            html_content += obj.to_html()
        else:
            html_content += f"<pre>{repr(obj)}</pre>"

    if "_last_expr_value" in local_ns:
        val = local_ns["_last_expr_value"]
        if all(val is not obj for obj in displayed_objects):
            if isinstance(val, plt.Figure):
                html_content += _fig_to_base64(val)
            elif "pandas" in str(type(val)):
                html_content += val.to_html()
            else:
                html_content += f"<pre>{repr(val)}</pre>"

    section_html = f"""
    <div style="display:grid; grid-template-columns:650px minmax(200px, 1fr); 
                column-gap:50px; row-gap:50px; align-items:start; margin-bottom:50px; width:100%;">
        <div><h4>{block_title}</h4>{html_content}</div>
        <div>
            <h4>Resources</h4>
            <p><b>CPU Time:</b> {cpu_sec:.3f} s</p>
            <p><b>Wall Time:</b> {wall_sec:.3f} s</p>
            <p><b>Memory Used:</b> {mem_mb:.2f} MB</p>
        </div>
    </div>
    """

    _inject("Code_outputs-container", section_html, report_file=report_file)

    # --- Track resource usage ---
    _cpu_times.append(cpu_sec)
    _wall_times.append(wall_sec)
    _mem_used.append(mem_mb)
    _plot_labels.append(block_title)



# --- Finalize resources ---
def finalize_resources(report_file=None):
    """Inject summary metrics into summary section and generate resource plots into resources section."""
    if not _cpu_times:
        return

    total_blocks = len(_plot_labels)
    total_cpu = sum(_cpu_times)
    total_wall = sum(_wall_times)
    total_mem = sum(_mem_used)

    metrics_html = f"""
    <p><b>Total Blocks:</b> {total_blocks} |
       <b>Total CPU Time:</b> {total_cpu:.3f} s |
       <b>Total Wall Time:</b> {total_wall:.3f} s |
       <b>Total Memory Used:</b> {total_mem:.2f} MB
    </p>
    """
    _inject("summary-metrics", metrics_html, report_file=report_file)

    for metric, values, color, title in zip(
        ["CPU Time (s)", "Wall Time (s)", "Memory Used (MB)"],
        [_cpu_times, _wall_times, _mem_used],
        ["tab:blue", "tab:red", "tab:green"],
        ["CPU Time per Block", "Wall Time per Block", "Memory Used per Block"]
    ):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(_plot_labels, values, color=color, alpha=0.7)
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.set_xticklabels(_plot_labels, rotation=45, ha="right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        _inject("resources-container", _fig_to_base64(fig), report_file=report_file)
        plt.close(fig)
