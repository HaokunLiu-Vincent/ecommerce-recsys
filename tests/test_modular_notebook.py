"""Validate the root Colab client's explicit stage and final-Test boundaries.

These tests inspect executable Notebook cells and run their orchestration with
recorded argv; they do not substitute model implementations or launch Colab.
The shared runtime's integration tests exercise actual bounded stage workers.
Inputs are the repository client and temporary log roots. No real data, cloud
service, secret, model fit or final-Test label is accessed by this module.
"""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "colab_training_val.ipynb"
STAGES = (
    "prepare",
    "trend",
    "repurchase",
    "sibling",
    "itemcf",
    "wals",
    "twotower",
    "fusion",
    "features",
    "ranker",
    "validation",
    "reload",
    "final_test",
)


def _cell(tag: str) -> str:
    """Read one actual code cell by its stable orchestration tag."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and tag in cell["metadata"].get("tags", [])
    )


def _settings(**overrides: str | bool | None) -> dict:
    """Execute real configuration after replacing only explicitly requested values."""
    tree = ast.parse(_cell("configuration"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in overrides:
                node.value = ast.Constant(overrides[name])
    namespace = {}
    exec(compile(ast.fix_missing_locations(tree), str(NOTEBOOK), "exec"), namespace)
    return namespace


def _stage_namespace(tmp_path: Path, action: str = "run") -> tuple[dict, list]:
    """Use the actual Notebook command wrapper with a no-subprocess argv recorder."""
    calls = []
    namespace = {
        "CLI": [sys.executable, str(ROOT / "scripts/run_pipeline.py")],
        "OPTIONS": ["--profile", "smoke", "--run-id", "notebook_fixture"],
        "RUN_ACTION": action,
        "LOG_ROOT": tmp_path,
        "REPORT_ROOT": tmp_path / "reports",
        "STAGE_RESULTS": {},
        "run_command": lambda argv, log: calls.append((argv, log)),
    }
    exec(compile(_cell("stage-wrapper"), str(NOTEBOOK), "exec"), namespace)
    return namespace, calls


def test_root_notebook_structure_and_safe_defaults() -> None:
    """Validate all code syntax, clear outputs and default smoke/Test isolation."""
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"{NOTEBOOK}:cell{index}", "exec")
            assert cell.execution_count is None and not cell.outputs
    settings = _settings()
    assert settings["RUN_PROFILE"] == "smoke"
    assert settings["RUN_ACTION"] == "run"
    assert settings["ALLOW_FULL_RUN"] is False
    assert settings["RUN_FINAL_TEST"] is False
    assert settings["ALLOW_FINAL_TEST"] is False
    assert settings["LOCKED_RUN_SIGNATURE"] is None
    assert settings["GIT_COMMIT"] is None
    assert settings["GITHUB_TOKEN_SECRET"] is None
    with pytest.raises(ValueError, match="smoke/full"):
        _settings(RUN_PROFILE="pilot")
    with pytest.raises(ValueError, match="explicit"):
        _settings(RUN_PROFILE="full")
    with pytest.raises(ValueError, match="simple"):
        _settings(RUN_ID="../wrong")
    _settings(RUN_PROFILE="full", ALLOW_FULL_RUN=True, DEVICE="cuda")


def test_exactly_one_active_notebook() -> None:
    """Exclude immutable archives/data/environments when counting active clients."""
    active = []
    excluded = {ROOT / "data", ROOT / ".venv", ROOT / ".git", ROOT / "docs/achive"}
    for base, directories, files in os.walk(ROOT):
        directories[:] = [
            name
            for name in directories
            if Path(base) / name not in excluded
            and name not in (".ipynb_checkpoints", "__pycache__")
        ]
        active.extend(Path(base) / name for name in files if name.endswith(".ipynb"))
    assert active == [NOTEBOOK]


def test_each_stage_has_one_ordered_dedicated_cell() -> None:
    """Bind every required stage exactly once and keep final Test last in code."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    observed = [
        tag.removeprefix("stage:")
        for cell in code_cells
        for tag in cell["metadata"].get("tags", [])
        if tag.startswith("stage:")
    ]
    assert observed == list(STAGES)
    assert "stage:final_test" in code_cells[-1]["metadata"]["tags"]
    for stage in STAGES:
        calls = [
            node
            for node in ast.walk(ast.parse(_cell("stage:" + stage)))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_stage"
        ]
        assert len(calls) == 1
        assert isinstance(calls[0].args[0], ast.Constant)
        assert calls[0].args[0].value == stage
    sources = "\n".join("".join(cell["source"]) for cell in code_cells)
    assert "--until" not in sources
    assert "zipfile" not in sources and "RELEASE_NAME" not in sources
    assert "hm_recsys.colab" not in sources and "hm_recsys.pilot" not in sources
    assert "hm_recsys.runtime.notebook" in sources
    assert "hm_recsys.runtime.git_checkout" in sources


@pytest.mark.parametrize("stage", STAGES[:-1])
@pytest.mark.parametrize("action", ("run", "resume", "status"))
def test_stage_cell_executes_only_requested_cli_stage(
    tmp_path: Path, stage: str, action: str
) -> None:
    """Exercise actual cell/wrapper code without implicitly invoking prerequisites."""
    namespace, calls = _stage_namespace(tmp_path, action)
    exec(compile(_cell("stage:" + stage), str(NOTEBOOK), "exec"), namespace)
    assert len(calls) == 1
    argv, log = calls[0]
    assert argv[2:7] == ["stage", "--stage", stage, "--stage-action", action]
    assert argv[7:] == ["--profile", "smoke", "--run-id", "notebook_fixture"]
    assert log == tmp_path / f"{stage}.log"
    assert set(namespace["STAGE_RESULTS"]) == {stage}
    assert not any(
        flag in argv for flag in ("--until", "--run-final-test", "--approve-final-test")
    )


def test_final_test_defaults_skip_and_separate_gates(tmp_path: Path, capsys) -> None:
    """Require Full, both approvals and an explicit frozen signature before argv."""
    namespace, calls = _stage_namespace(tmp_path)
    namespace.update(_settings())
    final_source = _cell("stage:final_test")
    exec(compile(final_source, str(NOTEBOOK), "exec"), namespace)
    assert not calls and "SKIPPED" in capsys.readouterr().out
    namespace["RUN_FINAL_TEST"] = True
    with pytest.raises(PermissionError, match="separate"):
        exec(compile(final_source, str(NOTEBOOK), "exec"), namespace)
    namespace.update(RUN_PROFILE="full", ALLOW_FINAL_TEST=True)
    with pytest.raises(ValueError, match="LOCKED_RUN_SIGNATURE"):
        exec(compile(final_source, str(NOTEBOOK), "exec"), namespace)
    assert not calls
    namespace["LOCKED_RUN_SIGNATURE"] = "a" * 64
    exec(compile(final_source, str(NOTEBOOK), "exec"), namespace)
    assert len(calls) == 1
    assert calls[0][0][-4:] == [
        "--run-final-test",
        "--approve-final-test",
        "--locked-signature",
        "a" * 64,
    ]
    assert calls[0][0][4] == "final_test"


def test_stage_failure_is_visible_and_not_marked_success(tmp_path: Path) -> None:
    """Propagate the real command error without inventing completion or retries."""
    namespace, calls = _stage_namespace(tmp_path)

    def fail(argv: list[str], log: Path) -> None:
        calls.append((argv, log))
        raise subprocess.CalledProcessError(9, argv)

    namespace["run_command"] = fail
    with pytest.raises(subprocess.CalledProcessError) as error:
        exec(compile(_cell("stage:wals"), str(NOTEBOOK), "exec"), namespace)
    assert error.value.returncode == 9
    assert len(calls) == 1 and calls[0][0][4] == "wals"
    assert not namespace["STAGE_RESULTS"]


def test_notebook_wrapper_streams_real_worker_events_before_exit(
    tmp_path: Path, monkeypatch
) -> None:
    """Run the actual Notebook progress wrapper around a bounded child process."""
    monkeypatch.setenv("PYTHONPATH", str(ROOT / "src"))
    monkeypatch.delenv("HM_PROGRESS_DISABLED", raising=False)
    namespace = _settings()
    namespace.update(PROJECT_ROOT=tmp_path, LOCAL_DATA_ROOT=tmp_path / "data")
    exec(compile(_cell("live-progress"), str(NOTEBOOK), "exec"), namespace)
    done = tmp_path / "worker_finished"
    events = []
    before_exit = []

    def observe(event: dict) -> None:
        events.append(event)
        before_exit.append(not done.exists())

    namespace["show_progress"] = observe
    payload = (
        "import time; from pathlib import Path; "
        "from hm_recsys.core.progress import emit; "
        "emit('batches',1,2,unit='batches',force=True,stage='twotower',"
        "metrics={'loss':0.75}); time.sleep(0.6); "
        "emit('batches',2,2,unit='batches',status='completed',stage='twotower',"
        "metrics={'loss':0.5}); "
        f"Path({str(done)!r}).touch()"
    )
    namespace["run_command"]([sys.executable, "-c", payload], tmp_path / "progress.log")
    assert [event["completed"] for event in events] == [1, 2]
    assert [event["percentage"] for event in events] == [50, 100]
    assert all(event["stage"] == "twotower" for event in events)
    assert events[0]["metrics"] == {"loss": 0.75}
    assert before_exit[0] and done.exists()


def test_actual_cli_exposes_modular_stage_contract() -> None:
    """Run real CLI help to verify every Notebook stage/approval flag is accepted."""
    child = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_pipeline.py"), "stage", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert child.returncode == 0, child.stderr
    for value in STAGES + (
        "--stage-action",
        "--run-final-test",
        "--approve-final-test",
        "--locked-signature",
        "--git-commit",
    ):
        assert value in child.stdout
