"""Validate useful module documentation and the README's complete active map.

The current standard requires informative module docstrings, English source
narrative and explicit file ownership. It deliberately does not require a
comment on every trivial statement. This module reads source/docs only; it does
not import archived code or execute models. Semantic prose review remains a
human responsibility rather than an arbitrary line-comment density score.
"""

import ast
from pathlib import Path

from hm_recsys.core.contracts import ROOT


def active_python_files() -> list[Path]:
    """Enumerate only maintained source, thin scripts and bounded test modules."""
    return (
        sorted((ROOT / "src").rglob("*.py"))
        + sorted((ROOT / "scripts").glob("*.py"))
        + sorted((ROOT / "tests").glob("*.py"))
    )


def test_active_modules_begin_with_specific_docstrings() -> None:
    """Require a real leading module docstring without mechanical comment rules."""
    seen = {}
    for path in active_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        doc = ast.get_docstring(tree)
        assert doc and doc.strip(), path.relative_to(ROOT)
        assert isinstance(tree.body[0], ast.Expr), path.relative_to(ROOT)
        if path.name != "__init__.py":
            assert doc not in seen, (path.relative_to(ROOT), seen.get(doc))
            seen[doc] = path.relative_to(ROOT)


def test_readme_maps_every_active_code_and_configuration_file() -> None:
    """Keep entrypoint/configuration roles discoverable after source moves."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    paths = active_python_files() + sorted((ROOT / "configs").glob("*.json"))
    paths += [ROOT / "colab_training_val.ipynb"]
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        assert f"]({relative})" in readme or f"`{relative}`" in readme, relative
    for package in (
        "core/",
        "data/",
        "retrieval/",
        "fusion/",
        "features/",
        "ranking/",
        "evaluation/",
        "runtime/",
    ):
        assert package in readme
