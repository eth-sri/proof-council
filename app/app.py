from __future__ import annotations

import argparse
import json
from pathlib import Path

from flask import Flask, abort, redirect, render_template, url_for


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUTS_ROOT = REPO_ROOT / "outputs"


parser = argparse.ArgumentParser()
parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUTS_ROOT))
parser.add_argument("--output-folder", type=str, default=None)
parser.add_argument("--port", type=int, default=5001)
parser.add_argument(
    "--disable-debug",
    action="store_true",
    help="Disable Flask debug mode.",
)
args = parser.parse_args()


def list_output_folders(output_root: Path) -> list[str]:
    if not output_root.exists():
        return []
    folders = [path.name for path in output_root.iterdir() if path.is_dir()]
    folders.sort(key=lambda name: (output_root / name).stat().st_mtime, reverse=True)
    return folders


def resolve_output_folder(output_root: Path, folder_name: str | None) -> tuple[str, Path]:
    folders = list_output_folders(output_root)
    if folder_name is None:
        if not folders:
            raise FileNotFoundError(f"No output folders found under {output_root}")
        folder_name = folders[0]
    folder_path = Path(folder_name)
    if not folder_path.is_absolute():
        folder_path = output_root / folder_name
    folder_path = folder_path.resolve()
    if not folder_path.exists():
        raise FileNotFoundError(f"Output folder does not exist: {folder_path}")
    try:
        display_name = str(folder_path.relative_to(output_root))
    except ValueError:
        display_name = folder_path.name
    return display_name, folder_path


OUTPUT_ROOT = Path(args.output_root).resolve()
CURRENT_FOLDER_NAME, CURRENT_FOLDER_PATH = resolve_output_folder(OUTPUT_ROOT, args.output_folder)


def folder_token(folder_name: str) -> str:
    return folder_name.replace("/", "---")


def folder_name_from_token(token: str) -> str:
    return token.replace("---", "/")


def load_results(folder_path: Path):
    grouped: dict[str, dict] = {}
    for json_path in sorted(folder_path.glob("*.json")):
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        problem_key = data.get("problem_name") or json_path.stem.split("__run_", 1)[0]
        run_idx = int(data.get("run_idx", 0))
        grouped.setdefault(
            problem_key,
            {
                "problem_name": problem_key,
                "problem": data.get("problem", ""),
                "problem_path": data.get("problem_path"),
                "runs": [],
                "runs_by_idx": {},
            },
        )
        run_data = {
            "file_name": json_path.name,
            "run_idx": run_idx,
            "conversation": data.get("conversation", []),
            "history": data.get("history", []),
            "detailed_cost": data.get("detailed_cost", {}),
            "final_response": data.get("final_response", ""),
            "created_at": data.get("created_at"),
            "config_ref": data.get("config_ref"),
        }
        grouped[problem_key]["runs"].append(run_data)
        grouped[problem_key]["runs_by_idx"][run_idx] = run_data

    for problem_key in grouped:
        grouped[problem_key]["runs"].sort(key=lambda run: run["run_idx"])
    return grouped


def get_sidebar_contents(current_problem: str | None = None):
    all_folders = list_output_folders(OUTPUT_ROOT)
    problems = {
        problem_key: {
            "name": problem_key,
            "ticks": f"({len(problem['runs'])} runs)",
            "class": "current" if problem_key == current_problem else "",
        }
        for problem_key, problem in sorted(RESULTS.items())
    }
    return {
        "dropdown": {"all_folders": all_folders, "current_folder": CURRENT_FOLDER_NAME},
        "reload_url": url_for("refresh", folder=folder_token(CURRENT_FOLDER_NAME)),
        "problems": problems,
    }


def render_message(message):
    role = message["role"]
    tagline = ""
    content = ""
    code = None
    is_cot = False

    if role == "developer":
        tagline = "System Prompt / Developer Message"
    elif role == "user":
        tagline = "User"
    elif role == "tool_response":
        tool_name = message.get("tool_name", "unknown")
        tool_call_id = message.get("tool_call_id")
        tagline = f"Response from Tool {tool_name} (Tool Call ID: {tool_call_id})"
    elif role == "assistant":
        msg_type = message.get("type")
        if msg_type == "cot":
            tagline = "Assistant (Chain-of-Thought)"
            is_cot = True
        elif msg_type == "response":
            tagline = "Assistant"
        elif msg_type == "tool_call":
            tool_name = message.get("tool_name", "unknown")
            tool_call_id = message.get("tool_call_id")
            tagline = f"Assistant (Tool Call to {tool_name}, Tool Call ID: {tool_call_id})"
            arguments = message.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            formatted_args = ""
            for key, value in arguments.items():
                if key == "code":
                    code = value
                else:
                    formatted_args += f"### {key}\n{value}\n\n"
            if formatted_args:
                content = formatted_args.strip()
        elif msg_type == "internal_tool_call":
            tool_name = message.get("tool_name", "unknown")
            tagline = f"Assistant (Internal Tool Call to {tool_name})"
            code = message.get("code")
        else:
            tagline = "Assistant"
    else:
        raise ValueError(f"Unknown role: {role}")

    if code is not None:
        code = str(code).replace("```python", "").replace("```", "").strip()

    if not content:
        content = message.get("content", "")

    if isinstance(content, list):
        text, img = None, None
        for item in content:
            if item["type"] in ["text", "input_text"]:
                text = item["text"]
            elif item["type"] == "input_image":
                img = item["image_url"]
            elif item["type"] == "image_url":
                img = item["image_url"]["url"]
        content = {"text": text, "img": img}
    else:
        content = {"text": str(content).strip(), "img": None}

    return {"tagline": tagline, "content": content, "code": code, "role": role, "is_cot": is_cot}


def render_conversation_html(conversation):
    messages_html = ""
    for index, message in enumerate(conversation):
        is_last_message = index == len(conversation) - 1
        msg_data = render_message(message)
        if (
            message.get("role") == "assistant"
            and not msg_data["content"]["text"]
            and not msg_data["content"]["img"]
            and not msg_data["code"]
            and not is_last_message
        ):
            continue
        messages_html += render_template("message.html", **msg_data)
    return messages_html


def get_instance_metadata(run_data):
    tool_calls = {}
    steps = run_data["history"] if run_data["history"] else [{"messages": run_data["conversation"]}]
    for step in steps:
        for message in step.get("messages", []):
            if "tool_call" in message.get("type", ""):
                tool_name = message.get("tool_name")
                if tool_name:
                    tool_calls[tool_name] = tool_calls.get(tool_name, 0) + 1
    return {"tool_calls": tool_calls, "cost": run_data["detailed_cost"]}


RESULTS = load_results(CURRENT_FOLDER_PATH)

app = Flask(__name__)


@app.route("/")
def index():
    total_runs = sum(len(problem["runs"]) for problem in RESULTS.values())
    total_cost = sum(
        run["detailed_cost"].get("cost", 0)
        for problem in RESULTS.values()
        for run in problem["runs"]
    )
    return render_template(
        "index.html",
        title="MathAgents Output Viewer",
        sidebar=get_sidebar_contents(),
        folder_name=CURRENT_FOLDER_NAME,
        nb_problems=len(RESULTS),
        total_runs=total_runs,
        total_cost=round(total_cost, 4),
        problems=sorted(RESULTS.items()),
    )


@app.route("/refresh/<folder>")
def refresh(folder):
    global CURRENT_FOLDER_NAME, CURRENT_FOLDER_PATH, RESULTS
    CURRENT_FOLDER_NAME, CURRENT_FOLDER_PATH = resolve_output_folder(OUTPUT_ROOT, folder_name_from_token(folder))
    RESULTS = load_results(CURRENT_FOLDER_PATH)
    return redirect(url_for("index"))


@app.route("/problem/<path:problem_key>")
def problem_view(problem_key):
    problem = RESULTS.get(problem_key)
    if problem is None:
        abort(404)

    instances = []
    for run in problem["runs"]:
        conversation_id = f"{problem_key}>>{run['run_idx']}"
        metadata = get_instance_metadata(run)
        if metadata["cost"].get("cost") is not None:
            metadata["cost"]["cost"] = round(metadata["cost"]["cost"], 4)
        if metadata["cost"].get("time") is not None:
            metadata["cost"]["time"] = round(metadata["cost"]["time"], 2)
        instances.append(
            {
                "run": run["run_idx"] + 1,
                "run_idx": run["run_idx"],
                "conversation_id": conversation_id,
                "history": run["history"],
                "metadata": metadata,
                "final_response": run["final_response"] or "No final response found.",
                "file_name": run["file_name"],
                "created_at": run["created_at"],
                "config_ref": run["config_ref"],
            }
        )

    return render_template(
        "problem.html",
        title="MathAgents Output Viewer",
        sidebar=get_sidebar_contents(current_problem=problem_key),
        folder_name=CURRENT_FOLDER_NAME,
        problem_name=problem_key,
        problem_statement=problem["problem"],
        instances=instances,
    )


@app.route("/modelinteraction/<path:identifier>")
def model_interaction(identifier):
    tokens = identifier.split(">>")
    if len(tokens) == 3 and tokens[2] == "history":
        problem_key = tokens[0]
        run_idx = int(tokens[1])
        run = RESULTS[problem_key]["runs_by_idx"][run_idx]
        options = []
        for step_idx, step in enumerate(run["history"]):
            label = f"TIME={step.get('timestep', step_idx)} 🕐 {step.get('step', f'Step {step_idx + 1}')}"
            value = f"{problem_key}>>{run_idx}>>{step_idx}"
            options.append(f'<option value="{value}">{label}</option>')
        target_id = f"history-step-content-{problem_key}-{run_idx}".replace("/", "--")
        return f"""
            <select onchange="loadHistoryStep(this.value, '{target_id}')" class="history-step-selector">
                <option value="">Select a step...</option>
                {''.join(options)}
            </select>
            <div id="{target_id}" class="history-step-content" style="margin-top: 1rem;"></div>
        """

    if len(tokens) != 2:
        abort(404)
    problem_key = tokens[0]
    run_idx = int(tokens[1])
    conversation = RESULTS[problem_key]["runs_by_idx"][run_idx]["conversation"]
    return render_conversation_html(conversation)


@app.route("/historystep/<path:identifier>")
def history_step(identifier):
    tokens = identifier.split(">>")
    if len(tokens) != 3:
        abort(404)
    problem_key = tokens[0]
    run_idx = int(tokens[1])
    step_idx = int(tokens[2])
    run = RESULTS[problem_key]["runs_by_idx"][run_idx]
    history = run["history"]
    if step_idx < 0 or step_idx >= len(history):
        abort(404)
    return render_conversation_html(history[step_idx].get("messages", []))


if __name__ == "__main__":
    app.run(port=args.port, debug=not args.disable_debug)
