"""The agent's tools: plain lookups and exact arithmetic over the practice project's documents. Each returns text,
which is what the AI model reads back."""
import ast
import csv
import math
import operator
import re
from pathlib import Path
from typing import Dict, List

# the inputs each tool takes (fixed); the descriptions the model reads are the notebook's wordings
PARAMETERS = {
    "search_spec": {"type": "object", "properties": {"query": {"type": "string", "description": "the search words, for example: foundation walls maximum w/cm"}},
                    "required": ["query"]},
    "calculator": {"type": "object", "properties": {"expression": {"type": "string", "description": "the arithmetic expression, for example: 310 / (500 + 120)"}},
                   "required": ["expression"]},
    "lookup_log": {"type": "object", "properties": {"log": {"type": "string", "enum": ["submittals", "schedule"], "description": "which log to read"},
                                                     "search": {"type": "string", "description": "the text to find, for example a submittal number or an element"}},
                   "required": ["log", "search"]},
    "load_skill": {"type": "object", "properties": {"name": {"type": "string", "description": "the skill's name"}}, "required": ["name"]},
}

STOP = {"the", "a", "an", "of", "for", "in", "on", "and", "or", "to", "is", "what", "this", "project", "project's", "with", "at", "by", "be", "must", "per"}


def _words(s: str) -> List[str]:
    return [w.rstrip("s") for w in re.findall(r"[a-z0-9/']+", s.lower()) if w not in STOP]


class Project:
    """The documents the tools read: the specification excerpt and the two logs."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.spec = (self.folder / "spec_03_30_00.txt").read_text(encoding="utf-8")
        self.logs = {"submittals": self._csv("submittal_log.csv"), "schedule": self._csv("schedule.csv")}

    def _csv(self, name):
        with open(self.folder / name, encoding="utf-8") as f:
            rows = list(csv.reader(f))
        return rows[0], rows[1:]

    # ---------------------------------------------------------------- the tools
    def search_spec(self, query: str = "", **_) -> str:
        lines = [l for l in self.spec.split("\n") if l.strip()]
        header = next((l for l in lines if l.startswith("Element")), "")
        q = set(_words(query))
        scored = sorted(((len(q & set(_words(l))), i, l) for i, l in enumerate(lines) if l != header), key=lambda t: (-t[0], t[1]))
        best = [l for s, i, l in scored[:3] if s > 0]
        if not best:
            return "No line of the specification matches those words."
        out = [l for l in lines if l in best]                      # in document order
        if any("|" in l for l in out) and header:
            out = [header] + out
        return "\n".join(out)

    def lookup_log(self, log: str = "", search: str = "", **_) -> str:
        if log not in self.logs:
            return f"There is no log called {log!r}; the logs are: submittals, schedule."
        head, rows = self.logs[log]
        s = search.lower().strip()
        hit = [r for r in rows if s and any(s in c.lower() for c in r)]
        if not hit:
            words = _words(search)
            hit = [r for r in rows if words and all(any(w in c.lower() for c in r) for w in words)]
        if not hit:
            return f"No row of the {log} log contains {search!r}. Its columns: " + " | ".join(head)
        return "\n".join([" | ".join(head)] + [" | ".join(r) for r in hit])

    @staticmethod
    def calculator(expression: str = "", **_) -> str:
        try:
            v = safe_eval(expression)
        except Exception as e:  # noqa: BLE001
            return f"Could not evaluate {expression!r}: {e}"
        return f"{expression} = {v:.6g}" if isinstance(v, float) else f"{expression} = {v}"


# ----------------------------------------------------------------------------- exact arithmetic, no Python beyond it
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv}
_FUN = {"ceil": math.ceil, "floor": math.floor, "round": round, "sqrt": math.sqrt, "min": min, "max": max, "abs": abs}


def safe_eval(expr: str):
    expr = expr.replace("×", "*").replace("÷", "/").replace("^", "**")
    expr = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", expr)            # 1,234 -> 1234 (min(1, 2) keeps its comma)
    node = ast.parse(expr.strip(), mode="eval").body

    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUN:
            return _FUN[n.func.id](*[ev(a) for a in n.args])
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "math" and n.attr in _FUN:
            return _FUN[n.attr]
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in _FUN:
            return _FUN[n.func.attr](*[ev(a) for a in n.args])
        raise ValueError("only numbers, + - * / ** ( ) and ceil, floor, round, sqrt, min, max are allowed")
    return ev(node)


def run_tool(project: Project, skills: Dict[str, str], name: str, args: dict) -> str:
    """Runs one tool call; any problem comes back as text, as a real tool would answer."""
    try:
        if name == "load_skill":
            n = str(args.get("name", "")).strip()
            return skills.get(n) or f"No skill called {n!r}. Available: " + ", ".join(skills)
        fn = {"search_spec": project.search_spec, "calculator": project.calculator, "lookup_log": project.lookup_log}.get(name)
        if fn is None:
            return f"There is no tool called {name!r}."
        return fn(**{k: str(v) for k, v in args.items()})
    except Exception as e:  # noqa: BLE001
        return f"The tool failed: {e}"
