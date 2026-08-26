"""Keep bundled notebooks aligned with the public package API."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parents[1] / "examples"
NOTEBOOKS = (
    EXAMPLES / "demo.ipynb",
    EXAMPLES / "intro_overview.ipynb",
    EXAMPLES / "profile_pipeline_speed.ipynb",
    EXAMPLES / "reconstruct_ecg.ipynb",
)


@pytest.mark.parametrize("notebook_path", NOTEBOOKS)
def test_notebook_code_cells_compile(notebook_path):
    notebook = json.loads(notebook_path.read_text())
    for cell_index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell.get("source", []))
        compile(source, f"{notebook_path.name}:cell-{cell_index}", "exec")


@pytest.mark.parametrize("notebook_path", NOTEBOOKS)
def test_notebooks_use_public_species_api(notebook_path):
    source = notebook_path.read_text()
    assert 'species=\\"human\\"' in source
    assert "sensitivity=" not in source
    assert "pyhearts._morphology" not in source
    assert "pyhearts.core.hybrid" not in source
    assert "hybrid_mod" not in source
