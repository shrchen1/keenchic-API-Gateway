from pathlib import Path

import pytest

import build_wheel


def test_http_logging_middleware_is_included_in_all_wheel_editions() -> None:
    specs = build_wheel.discover_descriptors()
    selected = {next(iter(specs)): specs[next(iter(specs))]}

    standard_plan = build_wheel.compile_plan(selected, edition="standard")
    taimide_plan = build_wheel.compile_plan(selected, edition="taimide")

    expected_module = "keenchic.core.http_logging"
    expected_path = "keenchic/core/http_logging.py"
    assert standard_plan.keenchic_cython[expected_module] == expected_path
    assert taimide_plan.keenchic_cython[expected_module] == expected_path


def test_grid_adapter_descriptor_packages_grid_proc() -> None:
    specs = build_wheel.discover_descriptors()
    spec = specs["ocr/meter-table-grid"]

    assert spec.adapter_source == "keenchic/inspections/adapters/ocr/meter_table_grid.py"
    assert any(
        entry.name == "procd_table_L"
        for submodule in spec.submodules
        for entry in submodule.bare
    )

    plan = build_wheel.compile_plan({spec.inspection_name: spec}, edition="standard")
    bare_extensions = {
        name
        for group in plan.bare_groups
        for name in group["extensions"]
    }
    assert "procd_table" in bare_extensions
    assert "procd_table_L" in bare_extensions


def test_grid_adapter_descriptor_includes_meter_table_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specs = build_wheel.discover_descriptors()
    spec = specs["ocr/meter-table-grid"]

    assert "keenchic/inspections/adapters/ocr/meter_table.py" in spec.dependencies

    plan = build_wheel.compile_plan({spec.inspection_name: spec}, edition="standard")
    assert "keenchic/inspections/adapters/ocr/meter_table.py" in plan.keep_py

    monkeypatch.setattr(build_wheel, "BUILD_DIR", tmp_path / "build")
    build_wheel.copy_to_staging(plan)
    assert (
        build_wheel.BUILD_DIR
        / "keenchic/inspections/adapters/ocr/meter_table.py"
    ).is_file()


def test_grid_adapter_taimide_plan_keeps_shared_dependency() -> None:
    specs = build_wheel.discover_descriptors()
    spec = specs["ocr/meter-table-grid"]

    plan = build_wheel.compile_plan({spec.inspection_name: spec}, edition="taimide")

    assert "keenchic/api/taimide_router.py" in plan.keep_py
    assert "keenchic/inspections/adapters/ocr/meter_table.py" in plan.keep_py
