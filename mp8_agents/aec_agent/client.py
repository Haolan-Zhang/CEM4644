"""The one place that talks to Gemini: single requests (a prompt, optionally with a drawing) and agent runs (the
model calls tools in a loop until it answers). Every request and every agent run is cached on disk, so the default
settings work without a key and the class does not hit the free tier's rate limits; a key is needed only for a
setting nobody ran before (an edited prompt, tool description or skill)."""
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .texts import MODEL

MAX_STEPS = 15                       # rounds of tool calls before the agent must answer
WAITS = (5, 10, 20, 40, 60)


def _key(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, ensure_ascii=False).encode()); h.update(b"\0")
    return h.hexdigest()[:20]


def file_key(path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def find_api_key(explicit: Optional[str] = None) -> Optional[str]:
    """The key typed in Step 0, else the GEMINI_API_KEY environment variable, else a Colab secret of that name."""
    if explicit and explicit.strip() and not explicit.strip().startswith("<"):
        return explicit.strip()
    k = os.environ.get("GEMINI_API_KEY", "").strip()
    if k:
        return k
    try:
        from google.colab import userdata
        return (userdata.get("GEMINI_API_KEY") or "").strip() or None
    except Exception:  # noqa: BLE001
        return None


def _text(r) -> str:
    try:
        return "".join(p.text or "" for p in (r.candidates[0].content.parts or []) if getattr(p, "text", None)) if r.candidates else ""
    except (AttributeError, IndexError, TypeError):
        return ""


class NoKey(RuntimeError):
    pass


class Gemini:
    def __init__(self, api_key: Optional[str], cache_dir: Path, model: str = MODEL, log=print):
        self.key = (api_key or "").strip() or None
        self.model = model
        self.cache = Path(cache_dir); self.cache.mkdir(parents=True, exist_ok=True)
        self.log = log
        self._sdk = None

    # ---------------------------------------------------------------- cache
    def _get(self, key):
        p = self.cache / f"{key}.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def _put(self, key, value):
        (self.cache / f"{key}.json").write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def _client(self):
        if not self.key:
            raise NoKey("This setting was not run before, and without a Gemini key it cannot run live: add your key in Step 0.")
        if self._sdk is None:
            import logging
            logging.getLogger("google_genai").setLevel(logging.ERROR); logging.getLogger("google_genai.models").setLevel(logging.ERROR)
            from google import genai
            self._sdk = genai.Client(api_key=self.key)
        return self._sdk

    def _generate(self, contents, config):
        """One request, with patience on the free tier's rate limits (wait and retry). Returns (response, seconds): the
        request's own time, without the waiting in between."""
        last = ""
        for attempt in range(len(WAITS) + 1):
            try:
                t0 = time.time()
                r = self._client().models.generate_content(model=self.model, contents=contents, config=config)
                return r, time.time() - t0
            except NoKey:
                raise
            except Exception as e:  # noqa: BLE001
                last = str(e).replace(self.key or "\0", "<key>")
                busy = any(s in last for s in ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE"))
                if busy and attempt < len(WAITS):
                    m = re.search(r"retry in (\d+(?:\.\d+)?)\s*s", last)
                    wait = max(WAITS[attempt], min(90, float(m.group(1)) + 1)) if m else WAITS[attempt]
                    self.log(f"   Gemini is busy; waiting {wait:.0f} s and trying again...")
                    time.sleep(wait); continue
                break
        msg = last.split("'message': '")[-1].split("'")[0] if "'message'" in last else last[:300]
        raise RuntimeError(f"Gemini request failed: {msg}")

    # ---------------------------------------------------------------- one request
    def ask(self, prompt: str, image: Optional[Path] = None) -> dict:
        """{text, seconds, cached}"""
        key = _key("ask", self.model, prompt, file_key(image) if image else None)
        hit = self._get(key)
        if hit:
            return {**hit, "cached": True}
        from google.genai import types
        parts = [types.Part(text=prompt)]
        if image:
            parts.insert(0, types.Part.from_bytes(data=Path(image).read_bytes(), mime_type="image/png"))
        cfg = types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_level="low"))
        r, dt = self._generate([types.Content(role="user", parts=parts)], cfg)
        out = {"text": r.text or "", "seconds": round(dt, 1)}
        self._put(key, out)
        return {**out, "cached": False}

    # ---------------------------------------------------------------- an agent run
    def agent(self, system: str, question: str, tools: Dict[str, dict], run_tool: Callable[[str, dict], str], extra=None) -> dict:
        """tools: {name: {"description", "parameters"}}. The model may call any of them, any number of times (up to
        MAX_STEPS rounds), and ends with a written answer. extra: anything else a tool's output depends on (a skill's body),
        so the cache knows it. Returns {steps: [{tool, args, output}], answer, seconds, model_calls, cached}."""
        key = _key("agent", self.model, system, question, {n: [t["description"], t["parameters"]] for n, t in sorted(tools.items())}, extra)
        hit = self._get(key)
        if hit:
            return {**hit, "cached": True}
        from google.genai import types
        decl = [types.FunctionDeclaration(name=n, description=t["description"], parameters_json_schema=t["parameters"]) for n, t in tools.items()]
        cfg = types.GenerateContentConfig(system_instruction=system, thinking_config=types.ThinkingConfig(thinking_level="low"),
                                          tools=[types.Tool(function_declarations=decl)] if decl else None,
                                          automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
        contents = [types.Content(role="user", parts=[types.Part(text=question)])]
        steps: List[dict] = []
        answer, seconds, n_requests = "", 0.0, 0
        for _ in range(MAX_STEPS):
            r, dt = self._generate(contents, cfg)
            seconds += dt; n_requests += 1
            calls = r.function_calls or []
            if not calls:
                answer = _text(r)
                for _ in range(2):                          # an empty reply happens now and then on the free tier: ask again
                    if answer.strip():
                        break
                    r, dt = self._generate(contents, cfg); seconds += dt; n_requests += 1
                    answer = _text(r)
                if r.function_calls:                        # the repeated request asked for a tool after all
                    calls = r.function_calls
                else:
                    break
            contents.append(r.candidates[0].content)
            parts = []
            for c in calls:
                args = dict(c.args or {})
                out = run_tool(c.name, args)
                steps.append({"tool": c.name, "args": args, "output": out})
                parts.append(types.Part.from_function_response(name=c.name, response={"result": out}))
            contents.append(types.Content(role="user", parts=parts))
        else:                                               # the step limit: one last request with the tools switched off
            contents.append(types.Content(role="user", parts=[types.Part(text="Stop calling tools and give your final answer now.")]))
            last = types.GenerateContentConfig(system_instruction=system, thinking_config=types.ThinkingConfig(thinking_level="low"),
                                               tools=cfg.tools, tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="NONE")),
                                               automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
            r, dt = self._generate(contents, last)
            seconds += dt; n_requests += 1
            answer = r.text or "(the agent reached the step limit without an answer)"
        out = {"steps": steps, "answer": answer.strip(), "seconds": round(seconds, 1), "model_calls": n_requests}
        self._put(key, out)
        return {**out, "cached": False}
