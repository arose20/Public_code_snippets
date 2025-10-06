# capture_utils/capture.py
import os
from datetime import datetime
from IPython.display import display
from IPython.utils.capture import capture_output
from IPython import get_ipython
from io import StringIO
import sys
import contextlib

ip = get_ipython()
captures = {}

@contextlib.contextmanager
def tee_output(stdout_buffer, stderr_buffer, display_live=True):
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

def run_and_capture_all(key, code, display_live=True):
    """
    Run a code block, optionally displaying outputs live while capturing everything.
    """
    stdout_buf, stderr_buf = StringIO(), StringIO()
    with tee_output(stdout_buf, stderr_buf, display_live=display_live):
        with capture_output(stdout=False, stderr=False, display=True) as cap:
            ip.run_cell(code, store_history=False)

    # Instead of modifying cap, store everything in a dict
    captures[key] = {
        "stdout": stdout_buf.getvalue(),
        "stderr": stderr_buf.getvalue(),
        "outputs": cap.outputs
    }

    if display_live:
        print(f"\n[✔] Capture complete: '{key}' ({len(cap.outputs)} rich outputs stored)")
    return captures[key]


def show_capture_all(key, section_title=None):
    cap = captures.get(key)
    if not cap:
        print(f"[!] No capture found for key '{key}'")
        return
    if section_title:
        print(f"\n=== {section_title} ===")

    if cap["stdout"].strip():
        print(cap["stdout"].strip())
    if cap["stderr"].strip():
        print("\n[stderr]:")
        print(cap["stderr"].strip())
    for out in cap["outputs"]:
        display(out)


def save_capture_to_html(key, output_file="capture_output.html", title=None, include_timestamp=True):
    """
    Save captured outputs from `captures[key]` to an HTML file.

    - include_timestamp: if True, appends current datetime to the section title.
    """
    if key not in captures:
        raise ValueError(f"No capture found for key '{key}'")

    cap = captures[key]

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") if include_timestamp else ""
    block_title = title if title else key
    if include_timestamp and timestamp:
        block_title = f"{block_title} — {timestamp}"

    # Separator for clarity
    separator_html = f"""
    <hr style="border-top:2px solid #888; margin:20px 0;">
    <h3>{block_title}</h3>
    """

    # Combine stdout, stderr, rich outputs
    html_content = "<pre>\n"
    if cap["stdout"]:
        html_content += f"stdout:\n{cap['stdout']}\n"
    if cap["stderr"]:
        html_content += f"\nstderr:\n{cap['stderr']}\n"
    html_content += "</pre>\n"

    for out in cap["outputs"]:
        if hasattr(out, 'data') and 'text/html' in out.data:
            html_content += out.data['text/html']
        elif hasattr(out, 'data') and 'image/png' in out.data:
            img_b64 = out.data['image/png']
            html_content += f'<img src="data:image/png;base64,{img_b64}">'
        elif hasattr(out, 'data') and 'text/plain' in out.data:
            html_content += f"<pre>{out.data['text/plain']}</pre>"

    final_html = separator_html + html_content

    # Append or create HTML file
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            content = f.read()
        if "</body>" in content:
            updated = content.replace("</body>", final_html + "\n</body>", 1)
        else:
            updated = content + final_html
    else:
        updated = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <title>Captured Outputs</title>
        </head>
        <body>
        {final_html}
        </body>
        </html>
        """

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(updated)

    print(f"[✔] Saved capture '{key}' to {output_file}")

