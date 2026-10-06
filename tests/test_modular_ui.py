"""The dashboard's local server: the API, its guards, and one real comparison run on Tiny."""

import json
import shutil
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from modular import server
from tests.conftest import ROOT


@pytest.fixture
def api(tmp_path):
    recipes = tmp_path / "recipes"
    shutil.copytree(ROOT / "modular" / "recipes", recipes)
    app = server.App(tmp_path / "agents", recipes)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), lambda *a: None)
    port = httpd.server_address[1]
    httpd.RequestHandlerClass = server.make_handler(app, port)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    def call(path, body=None, headers=None, method=None):
        data = None if body is None else json.dumps(body).encode()
        hdrs = {"Content-Type": "application/json"} if body is not None else {}
        hdrs |= headers or {}
        request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, headers=hdrs, method=method)
        try:
            with urllib.request.urlopen(request) as r:
                return r.status, json.loads(r.read() or b"null") if "json" in r.headers["Content-Type"] else r.read()
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    call.app, call.tmp = app, tmp_path
    yield call
    httpd.shutdown()
    httpd.server_close()


def good(name="my_agent"):
    return {"name": name, "base": "capacity", "atoms": [{"atom": "strait_open", "params": {"power": 2}}]}


def test_page_and_atoms(api):
    status, page = api("/")
    assert status == 200 and b"<title>" in page
    status, meta = api("/api/atoms")
    assert status == 200
    assert {a["name"] for a in meta["atoms"]} == {
        "strait_open", "fraction", "warning_cut", "demand_cap", "sanction_frontload",
    }  # fmt: skip
    assert [a["stage"] for a in meta["atoms"]] == sorted(
        (a["stage"] for a in meta["atoms"]), key=["allow", "scale", "stock"].index
    )
    assert {r["name"] for r in meta["recipes"]} >= set(server.SHIPPED)
    assert all(p["lo"] <= p["default"] <= p["hi"] for a in meta["atoms"] for p in a["params"])


def test_guards(api):
    assert api("/api/atoms", headers={"Host": "evil.example"})[0] == 403
    status, _ = api("/api/validate", good(), headers={"Content-Type": "text/plain"})
    assert status == 415
    assert api("/api/nope", {"recipe": good()})[0] == 404
    assert api("/api/upload", {"recipe": good()})[0] == 404  # there is no upload endpoint


def test_validate(api):
    status, out = api("/api/validate", {"recipe": good()})
    assert status == 200 and out["ok"] and out["source_lines"] > 100
    bad = good() | {"atoms": [{"atom": "strait_open", "params": {"power": 99}}]}
    status, out = api("/api/validate", {"recipe": bad})
    assert status == 400 and "outside" in out["error"]
    assert api("/api/validate", {"recipe": "nonsense"})[0] == 400


def test_build_writes_a_generated_agent_and_refuses_bad_names(api):
    status, out = api("/api/build", {"recipe": good()})
    assert status == 200 and (api.tmp / "agents" / "my_agent" / "agent.py").is_file()
    assert out["check"] == "uv run sbf check my_agent --task=small"
    for name in ("../escape", "Has Space", "", "9start", "x" * 41):
        assert api("/api/build", {"recipe": good(name)})[0] == 400
    assert not (api.tmp / "escape").exists()
    (api.tmp / "agents" / "hand").mkdir()
    (api.tmp / "agents" / "hand" / "agent.py").write_text("# mine\n")
    assert api("/api/build", {"recipe": good("hand")})[0] == 400
    assert (api.tmp / "agents" / "hand" / "agent.py").read_text() == "# mine\n"


def test_save_protects_shipped_recipes(api):
    assert api("/api/save", {"recipe": good("empty")})[0] == 400
    status, out = api("/api/save", {"recipe": good("mine_v1")})
    assert status == 200 and (api.tmp / "recipes" / "mine_v1.json").is_file()
    assert "mine_v1" in {r["name"] for r in api("/api/atoms")[1]["recipes"]}


def test_run_rejects_unknown_reference_and_name_clash(api):
    assert api("/api/run", {"recipe": good(), "references": ["nope"]})[0] == 400
    assert api("/api/run", {"recipe": good("empty"), "references": ["empty"]})[0] == 400
    assert api("/api/run", {"recipe": good(), "task": "full"})[0] == 400


def test_run_a_quick_comparison_on_tiny(api):
    status, out = api(
        "/api/run", {"recipe": good(), "task": "tiny", "mode": "quick", "ablate": True, "references": ["empty"]}
    )
    assert status == 200
    assert api("/api/run", {"recipe": good("other")})[0] == 409  # one run at a time, while this one is going
    deadline = time.time() + 300
    while time.time() < deadline:
        _, job = api(f"/api/job/{out['job']}")
        if job["status"] != "running":
            break
        time.sleep(1)
    assert job["status"] == "done", job["error"]
    names = [s["name"] for s in job["result"]["scores"]]
    assert names == ["my_agent", "empty"]
    assert [(a["recipe"], a["atom"]) for a in job["result"]["ablation"]] == [("my_agent", "strait_open")]
    assert all(s["fallback_weeks"] == 0 for s in job["result"]["scores"])


def test_neighbours_move_one_thing_each_and_stay_valid():
    from modular.compare import neighbours
    from modular.core import validate_recipe

    recipe = validate_recipe(good() | {"atoms": [{"atom": "demand_cap", "params": {"cover": 4, "slack": 1.5}}]})
    found = neighbours(recipe, limit=100)
    kinds = [k for _label, k, *_ in found]
    assert kinds.count("remove") == 1 and kinds.count("add") == 4 and kinds.count("param") == 4
    for _label, kind, atom, _param, _value, new in found:
        validate_recipe(new)  # every neighbour is a valid recipe, inside the parameters' bounds
        assert new["atoms"] != recipe["atoms"]


def test_suggest_job_ranks_neighbours_on_tiny(api):
    status, out = api("/api/run", {"recipe": good(), "kind": "suggest", "task": "tiny", "mode": "quick"})
    assert status == 200
    deadline = time.time() + 300
    while time.time() < deadline:
        _, job = api(f"/api/job/{out['job']}")
        if job["status"] != "running":
            break
        time.sleep(1)
    assert job["status"] == "done", job["error"]
    rows = job["result"]["suggestions"]
    assert rows and job["result"]["entropy"] == 12345  # tuned on a root of its own, not on dev
    diffs = [r["diff"] for r in rows]
    assert diffs == sorted(diffs, reverse=True)
    assert all(r["recipe"]["name"] == "my_agent" for r in rows)
    assert api("/api/run", {"recipe": good(), "kind": "bogus"})[0] == 400
