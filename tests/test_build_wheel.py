import json
import os
from datetime import date
from pathlib import Path
from zipfile import ZipFile

import pytest

import build_wheel


def _profile_spec(tmp_path: Path) -> build_wheel.AlgoSpec:
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    submodule = build_wheel.SubmoduleSpec(
        dir=tmp_path,
        weights_subdir="weights",
        engines=[
            build_wheel.EngineSpec(
                profile=build_wheel.TAIMIDE_JETSON_PROFILE,
                name="head",
                pattern="head-*.trt",
            ),
            build_wheel.EngineSpec(
                profile=build_wheel.TAIMIDE_JETSON_PROFILE,
                name="yolo",
                pattern="yolo-*.trt",
            ),
        ],
    )
    return build_wheel.AlgoSpec(
        inspection_name="ocr/test",
        descriptor_path=tmp_path / "test.build.toml",
        adapter_source="adapter.py",
        cython=False,
        dependencies=[],
        submodules=[submodule],
    )


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


@pytest.mark.parametrize(
    "value",
    ["2026.8.23", "2026.8.23.1", "2026.12.31.99"],
)
def test_release_version_accepts_calendar_versions(value: str) -> None:
    assert build_wheel.validate_release_version(value) == value


@pytest.mark.parametrize(
    "value",
    ["2026.8.23.0", "2026.02.30", "v2026.8.23", "2026.8"],
)
def test_release_version_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        build_wheel.validate_release_version(value)


def test_release_version_uses_system_date_when_omitted(tmp_path: Path) -> None:
    assert build_wheel.resolve_release_version(
        None,
        output_root=tmp_path,
        current_date=date(2026, 8, 25),
    ) == "2026.8.25"


def test_release_version_uses_next_highest_same_day_revision(
    tmp_path: Path,
) -> None:
    for name in (
        "2026.8.24.9",
        "2026.8.25",
        "2026.8.25.1",
        "2026.8.25.3",
        "2026.8.25.0",
        "unrelated",
    ):
        (tmp_path / name).mkdir()

    assert build_wheel.resolve_release_version(
        None,
        output_root=tmp_path,
        current_date=date(2026, 8, 25),
    ) == "2026.8.25.4"


def test_explicit_release_version_is_preserved(tmp_path: Path) -> None:
    (tmp_path / "2026.8.25").mkdir()

    assert build_wheel.resolve_release_version(
        "2026.8.24.2",
        output_root=tmp_path,
        current_date=date(2026, 8, 25),
    ) == "2026.8.24.2"


def test_profile_engines_select_latest_filename_date_not_mtime(tmp_path: Path) -> None:
    spec = _profile_spec(tmp_path)
    weights_dir = tmp_path / "weights"
    old_head = weights_dir / "head-20260801.trt"
    new_head = weights_dir / "head-20260821.trt"
    yolo = weights_dir / "yolo-20260823.trt"
    for path in (old_head, new_head, yolo):
        path.write_bytes(path.name.encode())

    os.utime(old_head, (2_000_000_000, 2_000_000_000))
    os.utime(new_head, (1_000_000_000, 1_000_000_000))

    selected = build_wheel.select_profile_engines(
        {spec.inspection_name: spec},
        build_wheel.TAIMIDE_JETSON_PROFILE,
    )

    assert selected["head"].path == new_head
    assert selected["head"].version_date == "20260821"
    assert selected["yolo"].path == yolo


def test_profile_engine_date_override_supports_rollback(tmp_path: Path) -> None:
    spec = _profile_spec(tmp_path)
    weights_dir = tmp_path / "weights"
    old_head = weights_dir / "head-20260801.trt"
    new_head = weights_dir / "head-20260821.trt"
    yolo = weights_dir / "yolo-20260823.trt"
    for path in (old_head, new_head, yolo):
        path.write_bytes(path.name.encode())

    selected = build_wheel.select_profile_engines(
        {spec.inspection_name: spec},
        build_wheel.TAIMIDE_JETSON_PROFILE,
        {"head": "20260801"},
    )

    assert selected["head"].path == old_head


def test_taimide_descriptors_declare_the_same_profile_engines() -> None:
    specs = build_wheel.discover_descriptors()
    declarations: list[set[tuple[str, str]]] = []

    for inspection_name in build_wheel.TAIMIDE_JETSON_ALGORITHMS:
        profile_engines = {
            (engine.name, engine.pattern)
            for submodule in specs[inspection_name].submodules
            for engine in submodule.engines
            if engine.profile == build_wheel.TAIMIDE_JETSON_PROFILE
        }
        declarations.append(profile_engines)

    assert declarations == [
        {
            ("head", "smp_Unet++_head_table_512-*.trt"),
            ("yolo", "yolo12_512_temper-*.trt"),
        },
        {
            ("head", "smp_Unet++_head_table_512-*.trt"),
            ("yolo", "yolo12_512_temper-*.trt"),
        },
    ]


def test_taimide_adapters_are_cython_protected() -> None:
    specs = build_wheel.discover_descriptors()
    selected = {
        name: specs[name]
        for name in build_wheel.TAIMIDE_JETSON_ALGORITHMS
    }

    assert all(spec.cython for spec in selected.values())

    plan = build_wheel.compile_plan(selected, edition="taimide")
    assert set(build_wheel.TAIMIDE_JETSON_PROTECTED_MODULES).issubset(
        plan.keenchic_cython
    )
    assert not {
        spec.adapter_source for spec in selected.values()
    }.intersection(plan.keep_py)


def test_taimide_profile_version_uses_only_taimide_local_tag() -> None:
    version = build_wheel._version_tag(
        list(build_wheel.TAIMIDE_JETSON_ALGORITHMS),
        {"ocr/datecode-num", *build_wheel.TAIMIDE_JETSON_ALGORITHMS},
        edition="taimide",
        base_version="2026.8.23.1",
        profile=build_wheel.TAIMIDE_JETSON_PROFILE,
    )

    assert version == "2026.8.23.1+taimide"


def test_taimide_profile_plan_packages_only_selected_engines() -> None:
    specs = build_wheel.discover_descriptors()
    selected = {
        name: specs[name]
        for name in build_wheel.TAIMIDE_JETSON_ALGORITHMS
    }
    weights_dir = (
        build_wheel.PROJECT_ROOT
        / "keenchic/inspections/ocr/temper_num_st/weights"
    )
    engines = {
        "head": build_wheel.SelectedEngine(
            name="head",
            profile=build_wheel.TAIMIDE_JETSON_PROFILE,
            pattern="smp_Unet++_head_table_512-*.trt",
            path=weights_dir / "smp_Unet++_head_table_512-20260821.trt",
            version_date="20260821",
            sha256="head",
        ),
        "yolo": build_wheel.SelectedEngine(
            name="yolo",
            profile=build_wheel.TAIMIDE_JETSON_PROFILE,
            pattern="yolo12_512_temper-*.trt",
            path=weights_dir / "yolo12_512_temper-20260823.trt",
            version_date="20260823",
            sha256="yolo",
        ),
    }

    plan = build_wheel.compile_plan(
        selected,
        edition="taimide",
        selected_engines=engines,
    )

    assert plan.weight_dirs == []
    assert plan.weight_files == [
        "keenchic/inspections/ocr/temper_num_st/weights/"
        "smp_Unet++_head_table_512-20260821.trt",
        "keenchic/inspections/ocr/temper_num_st/weights/"
        "yolo12_512_temper-20260823.trt",
    ]


def test_engine_override_parser_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicate engine override"):
        build_wheel.parse_engine_date_overrides(
            ["head=20260821", "head=20260807"]
        )


def test_wheel_archive_accepts_manifest_in_platform_wheel_purelib_data(
    tmp_path: Path,
) -> None:
    version = "2026.8.23+taimide"
    head = tmp_path / "head-20260821.trt"
    yolo = tmp_path / "yolo-20260823.trt"
    selected_engines = {
        "head": build_wheel.SelectedEngine(
            name="head",
            profile=build_wheel.TAIMIDE_JETSON_PROFILE,
            pattern="head-*.trt",
            path=head,
            version_date="20260821",
            sha256="head",
        ),
        "yolo": build_wheel.SelectedEngine(
            name="yolo",
            profile=build_wheel.TAIMIDE_JETSON_PROFILE,
            pattern="yolo-*.trt",
            path=yolo,
            version_date="20260823",
            sha256="yolo",
        ),
    }
    wheel_path = tmp_path / "test.whl"
    dist_info = f"keenchic_api_gateway-{version}.dist-info"
    data_dir = f"keenchic_api_gateway-{version}.data/purelib"
    with ZipFile(wheel_path, "w") as archive:
        archive.writestr(
            f"{dist_info}/METADATA",
            "Metadata-Version: 2.1\n"
            f"Version: {version}\n"
            f"Requires-Dist: pycuda (=={build_wheel.TAIMIDE_JETSON_PYCUDA})\n",
        )
        archive.writestr(
            f"{dist_info}/WHEEL",
            "Wheel-Version: 1.0\nTag: cp310-cp310-linux_aarch64\n",
        )
        archive.writestr(
            f"{data_dir}/keenchic/build_manifest.json",
            json.dumps({"package_version": version}),
        )
        archive.writestr(f"keenchic/weights/{head.name}", b"head")
        archive.writestr(f"keenchic/weights/{yolo.name}", b"yolo")
        for module_name in build_wheel.TAIMIDE_JETSON_PROTECTED_MODULES:
            module_path = module_name.replace(".", "/")
            archive.writestr(
                f"{module_path}.cpython-310-aarch64-linux-gnu.so",
                b"compiled",
            )

    build_wheel.validate_wheel_archive(
        wheel_path,
        version,
        selected_engines,
    )
