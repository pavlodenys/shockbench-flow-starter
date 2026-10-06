"""A local dashboard for building a policy from atoms: ``uv run python -m modular ui``.

Standard library only. It listens on 127.0.0.1, serves ``ui.html`` and a small JSON API: the atoms, validation, saving
a recipe, building an agent folder under ``agents/``, and a background comparison run. It has no upload endpoint:
nothing here talks to Codabench.
"""

import json
import re
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import modular.atoms  # noqa: F401 - registers the atoms
from modular import compare
from modular.build import render, write_agent
from modular.core import ATOMS, BASES, STAGES, RecipeError, validate_recipe


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SHIPPED = ("empty", "heuristic", "closure_cap", "full_stack")  # saved recipes may not overwrite these
NAME = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
STAGE_TITLES = {
    "allow": "Що дозволено",
    "scale": "Скільки відправляти",
    "stock": "Обмеження за запасом і попитом",
}


def atom_info():
    out = []
    for name, cls in ATOMS.items():
        lines = (cls.__doc__ or "").strip().splitlines()
        out.append(
            {
                "name": name,
                "stage": cls.stage,
                "summary": cls.ui or (lines[0] if lines else ""),
                "details": "" if cls.ui else " ".join(line.strip() for line in lines[1:]).strip(),
                "params": [
                    {"key": k, "default": p.default, "lo": p.lo, "hi": p.hi, "doc": p.doc}
                    for k, p in cls.params.items()
                ],
            }
        )
    out.sort(key=lambda a: (STAGES.index(a["stage"]), a["name"]))
    return out


class App:
    """The state the handlers share: folders to write to, and the one comparison job that may run at a time."""

    def __init__(self, agents_dir=None, recipes_dir=None):
        self.agents_dir = Path(agents_dir or ROOT / "agents")
        self.recipes_dir = Path(recipes_dir or HERE / "recipes")
        self.jobs = {}
        self.lock = threading.Lock()

    def recipes(self):
        out = []
        for path in sorted(self.recipes_dir.glob("*.json")):
            try:
                out.append(validate_recipe(json.loads(path.read_text())) | {"shipped": path.stem in SHIPPED})
            except (RecipeError, json.JSONDecodeError):
                continue
        return out

    def check_name(self, recipe):
        name = recipe.get("name") if isinstance(recipe, dict) else None
        if not isinstance(name, str) or not NAME.match(name):
            raise RecipeError(
                "ім'я агента: латиниця в нижньому регістрі, цифри та _, починається з літери, до 40 знаків"
            )
        return name

    def build(self, recipe, force=False):
        name = self.check_name(recipe)
        return write_agent(recipe, self.agents_dir / name, force)

    def save(self, recipe):
        name = self.check_name(recipe)
        if name in SHIPPED:
            raise RecipeError(f"{name!r} — приклад із комплекту; оберіть інше ім'я")
        clean = validate_recipe(recipe)
        (self.recipes_dir / f"{name}.json").write_text(json.dumps(clean, indent=2) + "\n")
        return name

    def start_job(self, body):
        recipe = body.get("recipe")
        self.check_name(recipe)
        task, mode, kind = body.get("task", "tiny"), body.get("mode", "quick"), body.get("kind", "compare")
        if kind not in ("compare", "suggest"):
            raise RecipeError("kind: compare або suggest")
        if task not in ("tiny", "small") or mode not in ("quick", "dev"):
            raise RecipeError("task: tiny або small; mode: quick або dev")
        recipes = [validate_recipe(recipe)]
        if kind == "suggest":
            body = {**body, "references": []}
        by_name = {r["name"]: r for r in self.recipes()}
        for ref in body.get("references", []):
            if ref not in by_name:
                raise RecipeError(f"немає рецепта {ref!r} для порівняння")
            if ref == recipes[0]["name"]:
                raise RecipeError(f"ім'я {ref!r} уже зайняте рецептом для порівняння: змініть ім'я агента")
            recipes.append(validate_recipe({k: v for k, v in by_name[ref].items() if k != "shipped"}))
        with self.lock:
            if any(j["status"] == "running" for j in self.jobs.values()):
                raise RuntimeError("уже йде інший прогін: дочекайтесь його кінця")
            job_id = uuid.uuid4().hex[:8]
            job = {"id": job_id, "status": "running", "log": [], "result": None, "error": None, "started": time.time()}
            self.jobs[job_id] = job

        def work():
            try:
                if kind == "suggest":
                    own = body.get("root", "own") == "own"  # tune on a root of your own, not on the dev episodes
                    job["result"] = compare.suggest(
                        recipes[0],
                        task=task,
                        episodes=(8 if mode == "quick" else 16) if own else "dev",
                        entropy=12345 if own else 0,
                        quick=(mode == "quick"),
                        log=lambda line: job["log"].append(line),
                    )
                    job["status"] = "done"
                    return
                job["result"] = compare.evaluate(
                    recipes,
                    task=task,
                    episodes="dev",
                    quick=(mode == "quick"),
                    ablate=[recipes[0]["name"]] if body.get("ablate") and recipes[0]["atoms"] else [],
                    log=lambda line: job["log"].append(line),
                )
                job["status"] = "done"
            except Exception as error:
                traceback.print_exc()
                job["error"], job["status"] = f"{type(error).__name__}: {error}", "error"

        threading.Thread(target=work, daemon=True).start()
        return job_id


def make_handler(app: App, port: int):
    page = (HERE / "ui.html").read_bytes()
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def _send(self, code, payload, content_type="application/json"):
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type + ("; charset=utf-8" if "text" in content_type else ""))
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _guard(self):
            if self.headers.get("Host") not in allowed_hosts:  # a page on another site cannot reach this server
                self._send(403, {"error": "forbidden host"})
                return False
            return True

        def do_GET(self):
            if not self._guard():
                return
            if self.path in ("/", "/index.html"):
                self._send(200, page, "text/html")
            elif self.path == "/api/atoms":
                self._send(
                    200, {"atoms": atom_info(), "stages": STAGE_TITLES, "bases": list(BASES), "recipes": app.recipes()}
                )
            elif self.path.startswith("/api/job/"):
                job = app.jobs.get(self.path.rsplit("/", 1)[-1])
                if job is None:
                    self._send(404, {"error": "no such job"})
                else:
                    self._send(200, {k: job[k] for k in ("id", "status", "log", "result", "error")})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self._guard():
                return
            if not (self.headers.get("Content-Type") or "").startswith("application/json"):
                return self._send(415, {"error": "application/json expected"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                recipe = body.get("recipe")
                if self.path == "/api/validate":
                    clean = validate_recipe(recipe)
                    self._send(200, {"ok": True, "recipe": clean, "source_lines": len(render(clean).splitlines())})
                elif self.path == "/api/build":
                    folder = app.build(recipe, force=bool(body.get("force")))
                    name = recipe["name"]
                    self._send(200, {"ok": True, "folder": folder, "check": f"uv run sbf check {name} --task=small"})
                elif self.path == "/api/save":
                    self._send(200, {"ok": True, "name": app.save(recipe)})
                elif self.path == "/api/run":
                    self._send(200, {"ok": True, "job": app.start_job(body)})
                else:
                    self._send(404, {"error": "not found"})
            except RecipeError as error:
                self._send(400, {"ok": False, "error": str(error)})
            except RuntimeError as error:
                self._send(409, {"ok": False, "error": str(error)})
            except (json.JSONDecodeError, AttributeError, TypeError, KeyError) as error:
                self._send(400, {"ok": False, "error": f"некоректний запит: {error}"})

    return Handler


def serve(port=8765, open_browser=True, agents_dir=None, recipes_dir=None):
    app = App(agents_dir, recipes_dir)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app, port))
    url = f"http://127.0.0.1:{port}/"
    print(f"конструктор агента: {url}  (Ctrl-C — зупинити; нічого не завантажується на Codabench)")
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
