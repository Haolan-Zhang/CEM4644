"""The one place that talks to Gemini. A request is an image plus a prompt (plus, optionally, a JSON schema the reply
must follow). Replies are cached on disk, so the built-in examples work without a key, and rate limits are handled
with patience (wait and retry) rather than errors."""
import hashlib
import io
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from .config import DEFAULT_MODEL, THINKING

MAX_SIDE = 1280


@dataclass
class Reply:
    text: str = ""                 # what the model wrote, verbatim
    data: Any = None               # the JSON parsed out of it (None if that failed)
    error: str = ""                # request error or parse error, in plain words
    seconds: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_thought: int = 0
    model: str = ""
    cached: bool = False
    schema: bool = False           # was a JSON schema enforced?

    @property
    def ok(self) -> bool:
        return self.error == "" and self.data is not None

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Reply":
        return Reply(**{k: v for k, v in d.items() if k in Reply.__dataclass_fields__})


def parse_json(text: str):
    """The JSON value inside a reply: strips ``` fences and any text around the first JSON value. Returns (data, error)."""
    s = (text or "").strip()
    s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
    s = re.sub(r"\s*```$", "", s).strip()
    if not s:
        return None, "empty reply"
    try:
        return json.loads(s), ""
    except json.JSONDecodeError as e:
        err = f"not valid JSON ({e.msg} at character {e.pos})"
    for o, c in (("[", "]"), ("{", "}")):
        i, j = s.find(o), s.rfind(c)
        if i != -1 and j > i:
            try:
                return json.loads(s[i:j + 1]), ""
            except json.JSONDecodeError:
                pass
    return None, err


def image_bytes(image: Image.Image, max_side: int = MAX_SIDE) -> bytes:
    im = image.convert("RGB")
    if max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=92)
    return buf.getvalue()


class ReplyCache:
    """data/cache/<variant>/<key>.json: one file per (model, prompt, schema, image, run)."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(model: str, prompt: str, schema: Optional[dict], img: bytes, run: int, image_key: Optional[str] = None) -> str:
        """image_key: a stable identity of a built-in example (the sha1 of its file on disk), so the key is the same on
        every machine; without one the JPEG bytes are hashed, which differs between Pillow builds (uploads only)."""
        h = hashlib.sha1()
        h.update(model.encode()); h.update(b"\0"); h.update(prompt.encode()); h.update(b"\0")
        h.update(json.dumps(schema, sort_keys=True).encode() if schema else b"-"); h.update(b"\0")
        h.update(image_key.encode() if image_key else hashlib.sha1(img).digest()); h.update(str(run).encode())
        return h.hexdigest()[:20]

    def get(self, key: str) -> Optional[Reply]:
        p = self.folder / f"{key}.json"
        if p.exists():
            r = Reply.from_dict(json.loads(p.read_text()))
            r.cached = True
            return r
        return None

    def put(self, key: str, reply: Reply):
        (self.folder / f"{key}.json").write_text(json.dumps(reply.to_dict(), ensure_ascii=False))

    def __len__(self):
        return len(list(self.folder.glob("*.json")))


class GeminiClient:
    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL, cache_dir: Optional[Path] = None, log=print):
        self.key = (api_key or "").strip() or None
        self.model = model
        self.cache = ReplyCache(cache_dir) if cache_dir else None
        self.log = log
        self.calls = 0
        self.seconds = 0.0
        self.tokens = 0
        self._client = None

    @property
    def live(self) -> bool:
        return bool(self.key)

    def _sdk(self):
        if self._client is None:
            import logging
            logging.getLogger("google_genai").setLevel(logging.ERROR)     # the SDK's advice about chat helpers is noise here
            logging.getLogger("google_genai.models").setLevel(logging.ERROR)
            from google import genai
            self._client = genai.Client(api_key=self.key)
        return self._client

    def ask(self, image: Optional[Image.Image], prompt: str, schema: Optional[dict] = None, thinking: str = THINKING,
            model: Optional[str] = None, use_cache: bool = True, run: int = 0, temperature: Optional[float] = None,
            image_key: Optional[str] = None) -> Reply:
        """One request. `schema` switches on the API's structured-output mode (the reply must follow it);
        without it, JSON is only what the prompt asks for. `image_key` identifies a built-in example for the cache."""
        model = model or self.model
        img = image_bytes(image) if image is not None else b""
        key = ReplyCache.key(model, prompt, schema, img, run, image_key)
        if use_cache and self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return hit
        if not self.live:
            return Reply(error="no API key: this request was not precomputed, and without a key it cannot run live. "
                               "Add your Gemini key in Step 0 to run it.", model=model, schema=schema is not None)
        from google.genai import types
        cfg = {}
        if schema is not None:
            cfg.update(response_mime_type="application/json", response_json_schema=schema)
        if thinking:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_level=thinking)
        if temperature is not None:
            cfg["temperature"] = temperature
        contents = [prompt] if image is None else [types.Part.from_bytes(data=img, mime_type="image/jpeg"), prompt]
        waits = (5, 10, 20, 40, 60)
        last = ""
        t0 = time.time()
        for attempt in range(len(waits) + 1):
            try:
                t0 = time.time()                      # the request's own time, without the waiting in between
                r = self._sdk().models.generate_content(model=model, contents=contents, config=types.GenerateContentConfig(**cfg))
                break
            except Exception as e:  # noqa: BLE001
                last = str(e)
                busy = "429" in last or "503" in last or "RESOURCE_EXHAUSTED" in last or "UNAVAILABLE" in last
                retry_in = re.search(r"retry in (\d+(?:\.\d+)?)\s*s", last)
                limit = re.search(r"limit: (\d+)", last)
                per_day = "429" in last and ("_per_day" in last or "PerDay" in last or (retry_in and float(retry_in.group(1)) > 300))
                if per_day:
                    return Reply(error=f"request failed: the daily free-tier quota of {model} is used up"
                                       + (f" (limit {limit.group(1)} requests per day)" if limit else "") + ". Pick another model in Step 0 or try again tomorrow.",
                                 seconds=time.time() - t0, model=model, schema=schema is not None)
                if busy and attempt < len(waits):
                    wait = waits[attempt]
                    if retry_in:
                        wait = max(wait, min(90, float(retry_in.group(1)) + 1))
                    self.log(f"   Gemini is busy ({'rate limit' if '429' in last else 'high demand'}"
                             + (f", {limit.group(1)} requests per minute" if limit and "429" in last else "") + f"); waiting {wait:.0f} s and trying again...")
                    time.sleep(wait)
                    continue
                msg = last.split("'message': '")[-1].split("'")[0] if "'message'" in last else last[:200]
                return Reply(error=f"request failed: {msg[:300]}", seconds=time.time() - t0, model=model, schema=schema is not None)
        else:
            return Reply(error="request failed: the model stayed busy for a minute or more; try again in a while.", seconds=time.time() - t0,
                         model=model, schema=schema is not None)
        dt = time.time() - t0
        text = r.text or ""
        u = getattr(r, "usage_metadata", None)
        data, err = parse_json(text)
        reply = Reply(text=text, data=data, error=err, seconds=round(dt, 2),
                      tokens_in=int(getattr(u, "prompt_token_count", 0) or 0), tokens_out=int(getattr(u, "candidates_token_count", 0) or 0),
                      tokens_thought=int(getattr(u, "thoughts_token_count", 0) or 0), model=model, schema=schema is not None)
        self.calls += 1; self.seconds += dt; self.tokens += reply.tokens_in + reply.tokens_out + reply.tokens_thought
        if self.cache is not None:
            self.cache.put(key, reply)
        return reply


def find_api_key(explicit: Optional[str] = None) -> Optional[str]:
    """The key typed in Step 0, else the GEMINI_API_KEY environment variable, else a Colab secret of that name."""
    if explicit and explicit.strip() and not explicit.strip().startswith("<"):
        return explicit.strip()
    k = os.environ.get("GEMINI_API_KEY", "").strip()
    if k:
        return k
    try:
        from google.colab import userdata
        k = (userdata.get("GEMINI_API_KEY") or "").strip()
        return k or None
    except Exception:
        return None



def file_key(path) -> str:
    """The sha1 of a file's bytes: the same on every machine, unlike a re-encoded image."""
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()
