"""``uv run python -m modular build recipe.json --out agents/<name>``; ``compare`` and ``atoms`` are the others."""

import sys

import fire

from modular import build as _build
from modular.core import ATOMS, RecipeError


def build(recipe: str, out: str | None = None, force: bool = False) -> None:
    """Generate the agent folder of a recipe (default agents/<recipe name>); never uploads anything."""
    folder = _build.build(recipe, out, force)
    print(f"written {folder}/agent.py: next, uv run sbf check {folder} --task=small")


def atoms() -> None:
    """List the atoms with their stage and parameters."""
    for name, cls in sorted(ATOMS.items(), key=lambda kv: (kv[1].stage, kv[0])):
        print(f"{name} [{cls.stage}]: {(cls.__doc__ or '').strip().splitlines()[0]}")
        for key, spec in cls.params.items():
            print(f"    {key} = {spec.default} in [{spec.lo}, {spec.hi}]  {spec.doc}")


def compare(
    *recipes: str,
    task: str = "small",
    episodes: str | int = "dev",
    entropy: int = 0,
    quick: bool = False,
    cpu_budget: bool = False,
    ablate: bool = False,
    n_jobs: int = -1,
) -> None:
    """Score recipes on the same episodes (and with --ablate, each atom's leave-one-out contribution).

    Args:
        recipes: recipe files.
        task: tiny, small or full.
        episodes: dev (the 20 public episodes), a count k (episodes 0..k-1) or a list.
        entropy: 0 for the dev root; any other integer for scenarios of your own (tune there, confirm on dev).
        quick: seconds, not the leaderboard's numbers.
        cpu_budget: play a week over the task's CPU budget as the naive rule, as the server does.
        ablate: also drop each atom in turn and print what it contributed.
        n_jobs: workers of a first run's reference computation.

    """
    from modular import compare as _compare

    _compare.run(recipes, task, episodes, entropy, quick, cpu_budget, ablate, n_jobs)


def ui(port: int = 8765, noopen: bool = False) -> None:
    """Open the local dashboard where a policy is assembled from atoms (127.0.0.1 only; never uploads)."""
    from modular import server

    server.serve(port, open_browser=not noopen)


def main() -> None:
    try:
        fire.Fire({"build": build, "atoms": atoms, "compare": compare, "ui": ui})
    except RecipeError as error:
        sys.exit(f"recipe error: {error}")


if __name__ == "__main__":
    main()
