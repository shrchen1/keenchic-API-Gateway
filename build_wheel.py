#!/usr/bin/env python3
"""Build keenchic-API-Gateway wheel with Cython-compiled .so files.

Compiles Python source to C++ shared libraries using Cython, then packages
everything (compiled .so, kept .py, model weights) into a .whl file.

Algorithm selection is driven by *.build.toml descriptor files co-located
with each adapter. Adding a new algorithm requires only a new descriptor —
this file needs no modification.

Usage (on Jetson Orin):
    pip install cython setuptools wheel numpy 'tomli; python_version < "3.11"'
    python3 build_wheel.py --profile taimide-jetson
    python3 build_wheel.py                              # all algorithms
    python3 build_wheel.py --list                       # list available
    python3 build_wheel.py -a ocr/datecode-num          # single algorithm
    python3 build_wheel.py -a ocr/datecode-num -a ocr/pill-count  # subset

Output:
    dist/<release>/keenchic_api_gateway-<release>+taimide-cp310-cp310-linux_aarch64.whl
    dist/<release>/{build-manifest.json,target-constraints.txt,SHA256SUMS}
    dist/keenchic_api_gateway-<version>-cp3<minor>-cp3<minor>-linux_aarch64.whl
    dist/keenchic_api_gateway-<version>+<algo_tag>-...-linux_aarch64.whl  (subset)
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        print("ERROR: tomli not installed. Run: pip install tomli")
        raise SystemExit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
BUILD_DIR = PROJECT_ROOT / "_build"
DIST_DIR = PROJECT_ROOT / "dist"

VERSION = "0.1.0"
BASE_PACKAGE_NAME = "keenchic-api-gateway"
TAIMIDE_JETSON_PROFILE = "taimide-jetson"
TAIMIDE_JETSON_ALGORITHMS = (
    "ocr/meter-table",
    "ocr/meter-table-grid",
)
TAIMIDE_JETSON_PROTECTED_MODULES = (
    "keenchic.inspections.adapters.ocr.meter_table",
    "keenchic.inspections.adapters.ocr.meter_table_grid",
)
TAIMIDE_JETSON_PYTHON = (3, 10)
TAIMIDE_JETSON_CUDA = "12.6"
TAIMIDE_JETSON_TENSORRT = "10.3"
TAIMIDE_JETSON_PYCUDA = "2026.1"
TAIMIDE_JETSON_L4T_RELEASE = "36"
TAIMIDE_JETSON_L4T_REVISION = "4.4"
TAIMIDE_JETSON_COMPUTE_CAPABILITY = (8, 7)

RELEASE_VERSION_RE = re.compile(
    r"^(?P<year>\d{4})\.(?P<month>\d{1,2})\.(?P<day>\d{1,2})(?:\.(?P<revision>[1-9]\d*))?$"
)
ENGINE_DATE_RE = re.compile(r"-(?P<date>\d{8})\.trt$")

DESCRIPTOR_GLOB = "keenchic/inspections/adapters/**/*.build.toml"

# Core modules always included regardless of algorithm selection
CORE_CYTHON: dict[str, str] = {
    "keenchic.core.inspection_manager": "keenchic/core/inspection_manager.py",
    "keenchic.core.http_logging":       "keenchic/core/http_logging.py",
    "keenchic.core.logging":            "keenchic/core/logging.py",
    "keenchic.inspections.base":        "keenchic/inspections/base.py",
    "keenchic.inspections.registry":    "keenchic/inspections/registry.py",
    "keenchic.inspections.result_codes": "keenchic/inspections/result_codes.py",
    "keenchic.api.deps":                "keenchic/api/deps.py",
    "keenchic.services.permit_lookup":  "keenchic/services/permit_lookup.py",
}

CORE_KEEP_PY: list[str] = [
    "main.py",
    "serve.py",
    "keenchic/__init__.py",
    "keenchic/core/__init__.py",
    "keenchic/core/config.py",
    "keenchic/core/file_saver.py",
    "keenchic/api/__init__.py",
    "keenchic/api/router.py",
    "keenchic/schemas/__init__.py",
    "keenchic/schemas/response.py",
    "keenchic/inspections/__init__.py",
    "keenchic/inspections/adapters/__init__.py",
    "keenchic/inspections/adapters/ocr/__init__.py",
    "keenchic/services/__init__.py",
]

# Extra files included only in the taimide edition
TAIMIDE_KEEP_PY: list[str] = [
    "keenchic/api/taimide_router.py",
]

# Runtime dependencies (excluding openvino — not supported on aarch64)
# Pre-installed on Jetson (JetPack 6.x), NOT listed here:
#   matplotlib, opencv-python, tensorrt
INSTALL_REQUIRES: list[str] = [
    "fastapi>=0.119.0",
    "numpy<2.0.0",
    "pycuda>=2026.1",
    "pydantic-settings>=2.0.0",
    "python-multipart>=0.0.9",
    "scikit-image>=0.25.0",
    "scikit-learn>=1.4.0",
    "structlog>=24.0.0",
    "uvicorn[standard]>=0.37.0",
]

TAIMIDE_JETSON_INSTALL_REQUIRES: list[str] = [
    requirement if not requirement.startswith("pycuda") else f"pycuda=={TAIMIDE_JETSON_PYCUDA}"
    for requirement in INSTALL_REQUIRES
]


# ---------------------------------------------------------------------------
# Descriptor data model
# ---------------------------------------------------------------------------

@dataclass
class SubmoduleEntry:
    name: str
    src: str


@dataclass(frozen=True)
class EngineSpec:
    profile: str
    name: str
    pattern: str


@dataclass(frozen=True)
class SelectedEngine:
    name: str
    profile: str
    pattern: str
    path: Path
    version_date: str
    sha256: str


@dataclass
class SubmoduleSpec:
    dir: Path
    dotted: list[SubmoduleEntry] = field(default_factory=list)
    bare: list[SubmoduleEntry] = field(default_factory=list)
    weights_subdir: str | None = None
    engines: list[EngineSpec] = field(default_factory=list)

    @property
    def weights_path(self) -> Path | None:
        if self.weights_subdir:
            return self.dir / self.weights_subdir
        return None


@dataclass
class AlgoSpec:
    inspection_name: str
    descriptor_path: Path
    adapter_source: str
    cython: bool
    dependencies: list[str]
    submodules: list[SubmoduleSpec]


# ---------------------------------------------------------------------------
# Build plan
# ---------------------------------------------------------------------------

@dataclass
class CompilePlan:
    keenchic_cython: dict[str, str]
    dotted_groups: list[dict]
    bare_groups: list[dict]
    keep_py: list[str]
    weight_dirs: list[str]
    weight_files: list[str]
    init_dirs: list[str]


# ---------------------------------------------------------------------------
# Discovery and selection
# ---------------------------------------------------------------------------

def discover_descriptors() -> dict[str, AlgoSpec]:
    """Load all *.build.toml descriptors; return {inspection_name: AlgoSpec}."""
    specs: dict[str, AlgoSpec] = {}
    for toml_path in sorted(PROJECT_ROOT.glob(DESCRIPTOR_GLOB)):
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)

        name: str = data.get("inspection_name", "")
        if not name:
            print(f"ERROR: {toml_path} is missing 'inspection_name'")
            sys.exit(1)

        adapter_cfg = data.get("adapter", {})
        adapter_src: str = adapter_cfg.get("source", "")
        cython: bool = adapter_cfg.get("cython", True)
        dependencies: list[str] = adapter_cfg.get("dependencies", [])

        if not adapter_src or not (PROJECT_ROOT / adapter_src).exists():
            print(f"ERROR: [{name}] adapter source not found: {adapter_src!r}")
            sys.exit(1)
        missing_dependencies = [
            path for path in dependencies if not (PROJECT_ROOT / path).is_file()
        ]
        if missing_dependencies:
            print(
                f"ERROR: [{name}] adapter dependencies not found: "
                f"{', '.join(missing_dependencies)}"
            )
            sys.exit(1)

        submodules: list[SubmoduleSpec] = []
        for sm in data.get("submodule", []):
            sm_dir = PROJECT_ROOT / sm["dir"]
            if not sm_dir.is_dir():
                print(f"ERROR: [{name}] submodule dir not found: {sm['dir']!r}")
                sys.exit(1)
            submodules.append(SubmoduleSpec(
                dir=sm_dir,
                dotted=[SubmoduleEntry(e["name"], e["src"]) for e in sm.get("dotted", [])],
                bare=[SubmoduleEntry(e["name"], e["src"]) for e in sm.get("bare", [])],
                weights_subdir=sm.get("weights_subdir"),
                engines=[
                    EngineSpec(
                        profile=e["profile"],
                        name=e["name"],
                        pattern=e["pattern"],
                    )
                    for e in sm.get("engine", [])
                ],
            ))

        specs[name] = AlgoSpec(
            inspection_name=name,
            descriptor_path=toml_path,
            adapter_source=adapter_src,
            cython=cython,
            dependencies=dependencies,
            submodules=submodules,
        )

    if not specs:
        print(f"ERROR: no descriptors found matching {DESCRIPTOR_GLOB!r}")
        sys.exit(1)

    return specs


def select_algorithms(specs: dict[str, AlgoSpec], cli_names: list[str]) -> dict[str, AlgoSpec]:
    """Filter specs by CLI names; empty list returns all."""
    if not cli_names:
        return specs

    invalid = [n for n in cli_names if n not in specs]
    if invalid:
        print(f"ERROR: Unknown algorithm(s): {', '.join(invalid)}")
        print(f"Available: {', '.join(sorted(specs))}")
        sys.exit(1)

    return {n: specs[n] for n in cli_names}


def validate_release_version(value: str) -> str:
    """Validate the YYYY.M.D[.N] release version used by the Jetson profile."""
    match = RELEASE_VERSION_RE.fullmatch(value)
    if match is None:
        raise ValueError(
            "release version must use YYYY.M.D or YYYY.M.D.N with N >= 1"
        )

    try:
        datetime(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
        )
    except ValueError as exc:
        raise ValueError(f"invalid release date: {value}") from exc

    return value


def resolve_release_version(
    value: str | None,
    output_root: Path = DIST_DIR,
    current_date: date | None = None,
) -> str:
    """Use an explicit version or select the next revision for the system date."""
    if value is not None:
        return validate_release_version(value)

    release_date = current_date or datetime.now().date()
    base_version = (
        f"{release_date.year}.{release_date.month}.{release_date.day}"
    )
    same_day_pattern = re.compile(
        rf"^{re.escape(base_version)}(?:\.(?P<revision>[1-9]\d*))?$"
    )
    existing_revisions: list[int] = []
    if output_root.exists():
        for candidate in output_root.iterdir():
            match = same_day_pattern.fullmatch(candidate.name)
            if match is not None:
                revision = match.group("revision")
                existing_revisions.append(int(revision) if revision else 0)

    if not existing_revisions:
        return base_version
    return f"{base_version}.{max(existing_revisions) + 1}"


def parse_engine_date_overrides(values: list[str]) -> dict[str, str]:
    """Parse repeated NAME=YYYYMMDD engine date overrides."""
    overrides: dict[str, str] = {}
    for value in values:
        name, separator, date_value = value.partition("=")
        if not separator or not name or not re.fullmatch(r"\d{8}", date_value):
            raise ValueError(
                f"invalid engine override {value!r}; expected NAME=YYYYMMDD"
            )
        try:
            datetime.strptime(date_value, "%Y%m%d")
        except ValueError as exc:
            raise ValueError(f"invalid engine date in override: {value!r}") from exc
        if name in overrides:
            raise ValueError(f"duplicate engine override: {name}")
        overrides[name] = date_value
    return overrides


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _engine_date(path: Path) -> str:
    match = ENGINE_DATE_RE.search(path.name)
    if match is None:
        raise ValueError(
            f"engine filename must end in -YYYYMMDD.trt: {path.name}"
        )
    date_value = match.group("date")
    try:
        datetime.strptime(date_value, "%Y%m%d")
    except ValueError as exc:
        raise ValueError(f"invalid engine date in filename: {path.name}") from exc
    return date_value


def select_profile_engines(
    selected: dict[str, AlgoSpec],
    profile: str,
    overrides: dict[str, str] | None = None,
) -> dict[str, SelectedEngine]:
    """Select one newest filename-versioned engine per profile engine name."""
    overrides = overrides or {}
    declared: dict[str, tuple[Path, EngineSpec]] = {}

    for spec in selected.values():
        for submodule in spec.submodules:
            for engine in submodule.engines:
                if engine.profile != profile:
                    continue
                if submodule.weights_path is None:
                    raise ValueError(
                        f"[{spec.inspection_name}] engine {engine.name!r} "
                        "requires weights_subdir"
                    )
                declaration = (submodule.weights_path, engine)
                previous = declared.get(engine.name)
                if previous is not None and previous != declaration:
                    raise ValueError(
                        f"conflicting declarations for engine {engine.name!r}"
                    )
                declared[engine.name] = declaration

    if not declared:
        raise ValueError(f"no engines declared for profile {profile!r}")

    unknown_overrides = sorted(set(overrides) - set(declared))
    if unknown_overrides:
        raise ValueError(
            f"unknown engine override(s): {', '.join(unknown_overrides)}"
        )

    selected_engines: dict[str, SelectedEngine] = {}
    for name, (weights_path, engine) in sorted(declared.items()):
        candidates_by_date: dict[str, list[Path]] = {}
        for candidate in sorted(weights_path.glob(engine.pattern)):
            date_value = _engine_date(candidate)
            candidates_by_date.setdefault(date_value, []).append(candidate)

        if not candidates_by_date:
            raise ValueError(
                f"no engine matches {engine.pattern!r} in {weights_path}"
            )

        selected_date = overrides.get(name, max(candidates_by_date))
        matching = candidates_by_date.get(selected_date, [])
        if not matching:
            raise ValueError(
                f"engine {name!r} has no candidate dated {selected_date}"
            )
        if len(matching) != 1:
            names = ", ".join(path.name for path in matching)
            raise ValueError(
                f"engine {name!r} has conflicting candidates dated "
                f"{selected_date}: {names}"
            )

        path = matching[0]
        selected_engines[name] = SelectedEngine(
            name=name,
            profile=profile,
            pattern=engine.pattern,
            path=path,
            version_date=selected_date,
            sha256=_sha256(path),
        )

    return selected_engines


def compile_plan(
    selected: dict[str, AlgoSpec],
    edition: str = "standard",
    selected_engines: dict[str, SelectedEngine] | None = None,
) -> CompilePlan:
    """Derive a CompilePlan from the selected AlgoSpecs."""
    keenchic_cython = dict(CORE_CYTHON)
    keep_py = list(CORE_KEEP_PY)
    if edition == "taimide":
        keep_py.extend(TAIMIDE_KEEP_PY)
    weight_dirs: list[str] = []
    weight_files: list[str] = []

    dotted_seen: set[tuple[str, str]] = set()
    bare_seen: set[tuple[str, str]] = set()
    dotted_by_cwd: dict[str, dict[str, str]] = {}
    bare_by_cwd: dict[str, dict[str, str]] = {}
    init_dirs_set: set[str] = set()

    for spec in selected.values():
        dotted_module_name = spec.adapter_source.replace("/", ".").removesuffix(".py")
        if spec.cython:
            keenchic_cython[dotted_module_name] = spec.adapter_source
        else:
            keep_py.append(spec.adapter_source)
        for dependency in spec.dependencies:
            if dependency not in keep_py:
                keep_py.append(dependency)

        for sm in spec.submodules:
            dir_rel = sm.dir.relative_to(PROJECT_ROOT)
            dir_str = str(dir_rel)
            parent_str = str(dir_rel.parent)

            init_dirs_set.add(parent_str)
            init_dirs_set.add(dir_str)

            # Dotted modules: compiled from parent dir (e.g. ocr/) so the
            # .so lands inside the submodule package dir.
            for entry in sm.dotted:
                key = (parent_str, entry.name)
                if key not in dotted_seen:
                    dotted_seen.add(key)
                    src_from_parent = dir_rel.name + "/" + entry.src
                    dotted_by_cwd.setdefault(parent_str, {})[entry.name] = src_from_parent

            # Bare modules: compiled from the submodule dir itself.
            for entry in sm.bare:
                key = (dir_str, entry.name)
                if key not in bare_seen:
                    bare_seen.add(key)
                    bare_by_cwd.setdefault(dir_str, {})[entry.name] = entry.src

            if selected_engines is not None:
                for engine in selected_engines.values():
                    if engine.path.parent == sm.weights_path:
                        engine_rel = str(engine.path.relative_to(PROJECT_ROOT))
                        if engine_rel not in weight_files:
                            weight_files.append(engine_rel)
            elif sm.weights_path and sm.weights_path.is_dir():
                w_rel = str(sm.weights_path.relative_to(PROJECT_ROOT))
                if w_rel not in weight_dirs:
                    weight_dirs.append(w_rel)

    compiled_sources = set(keenchic_cython.values())
    keep_py = [path for path in keep_py if path not in compiled_sources]

    return CompilePlan(
        keenchic_cython=keenchic_cython,
        dotted_groups=[{"cwd": cwd, "extensions": exts} for cwd, exts in dotted_by_cwd.items()],
        bare_groups=[{"cwd": cwd, "extensions": exts} for cwd, exts in bare_by_cwd.items()],
        keep_py=keep_py,
        weight_dirs=weight_dirs,
        weight_files=sorted(weight_files),
        init_dirs=sorted(init_dirs_set),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: list[str], **kwargs) -> None:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"required package is not installed: {name}") from exc


def validate_taimide_jetson_env() -> dict[str, object]:
    """Validate the fixed Build Jetson runtime profile."""
    errors: list[str] = []

    for package in ("Cython", "setuptools", "wheel", "numpy"):
        _distribution_version(package)

    if platform.system() != "Linux":
        errors.append(f"expected Linux, got {platform.system()}")
    if platform.machine() != "aarch64":
        errors.append(f"expected aarch64, got {platform.machine()}")
    if sys.version_info[:2] != TAIMIDE_JETSON_PYTHON:
        errors.append(
            "expected Python "
            f"{TAIMIDE_JETSON_PYTHON[0]}.{TAIMIDE_JETSON_PYTHON[1]}, "
            f"got {sys.version_info.major}.{sys.version_info.minor}"
        )

    l4t_text = Path("/etc/nv_tegra_release").read_text(errors="replace")
    expected_l4t = (
        f"# R{TAIMIDE_JETSON_L4T_RELEASE} (release), "
        f"REVISION: {TAIMIDE_JETSON_L4T_REVISION}"
    )
    if expected_l4t not in l4t_text:
        errors.append(
            f"expected L4T R{TAIMIDE_JETSON_L4T_RELEASE}."
            f"{TAIMIDE_JETSON_L4T_REVISION}"
        )

    nvcc_path = Path("/usr/local/cuda/bin/nvcc")
    if not nvcc_path.is_file():
        errors.append(f"CUDA compiler not found: {nvcc_path}")
        nvcc_output = ""
    else:
        nvcc_output = subprocess.run(
            [str(nvcc_path), "--version"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if f"release {TAIMIDE_JETSON_CUDA}" not in nvcc_output:
            errors.append(f"expected CUDA {TAIMIDE_JETSON_CUDA}")

    tensorrt_version = _distribution_version("tensorrt")
    if not tensorrt_version.startswith(f"{TAIMIDE_JETSON_TENSORRT}."):
        errors.append(
            f"expected TensorRT {TAIMIDE_JETSON_TENSORRT}.x, "
            f"got {tensorrt_version}"
        )

    pycuda_version = _distribution_version("pycuda")
    if pycuda_version != TAIMIDE_JETSON_PYCUDA:
        errors.append(
            f"expected PyCUDA {TAIMIDE_JETSON_PYCUDA}, got {pycuda_version}"
        )

    import pycuda.driver as cuda  # type: ignore[import]

    cuda.init()
    if cuda.Device.count() != 1:
        errors.append(f"expected exactly one CUDA device, got {cuda.Device.count()}")
        device_name = ""
        compute_capability: tuple[int, int] | tuple[()] = ()
    else:
        device = cuda.Device(0)
        device_name = device.name()
        compute_capability = device.compute_capability()
        if "orin" not in device_name.lower():
            errors.append(f"expected an Orin GPU, got {device_name}")
        if compute_capability != TAIMIDE_JETSON_COMPUTE_CAPABILITY:
            errors.append(
                "expected compute capability "
                f"{TAIMIDE_JETSON_COMPUTE_CAPABILITY}, got {compute_capability}"
            )

    if errors:
        raise RuntimeError("Taimide Jetson profile mismatch:\n- " + "\n- ".join(errors))

    return {
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "l4t": (
            f"R{TAIMIDE_JETSON_L4T_RELEASE}."
            f"{TAIMIDE_JETSON_L4T_REVISION}"
        ),
        "cuda": TAIMIDE_JETSON_CUDA,
        "tensorrt": tensorrt_version,
        "pycuda": pycuda_version,
        "gpu": device_name,
        "compute_capability": list(compute_capability),
    }


def validate_selected_engines(
    selected_engines: dict[str, SelectedEngine],
) -> None:
    """Deserialize selected TensorRT engines using the Build Jetson GPU stack."""
    import pycuda.driver as cuda  # type: ignore[import]
    import tensorrt as trt  # type: ignore[import]

    cuda.init()
    context = cuda.Device(0).make_context()
    loaded_engines: list[object] = []
    runtime = None
    logger = None
    try:
        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)
        for engine in selected_engines.values():
            serialized = engine.path.read_bytes()
            deserialized = runtime.deserialize_cuda_engine(serialized)
            if deserialized is None:
                raise RuntimeError(
                    f"TensorRT could not deserialize engine: {engine.path}"
                )
            loaded_engines.append(deserialized)
            deserialized = None
    finally:
        loaded_engines.clear()
        runtime = None
        logger = None
        gc.collect()
        context.pop()
        context.detach()


def _write_compile_setup(dest: Path, extensions: dict[str, str]) -> Path:
    """Write a temporary setup.py for Cython compilation."""
    ext_lines = []
    for mod_name, src_path in extensions.items():
        ext_lines.append(
            f'        Extension("{mod_name}", ["{src_path}"], language="c++"),'
        )
    ext_block = "\n".join(ext_lines)

    content = (
        "from setuptools import setup, Extension\n"
        "from Cython.Build import cythonize\n"
        "\n"
        "setup(\n"
        "    ext_modules=cythonize([\n"
        f"{ext_block}\n"
        '    ], compiler_directives={"language_level": "3"}),\n'
        ")\n"
    )
    setup_file = dest / "_cython_build.py"
    setup_file.write_text(content)
    return setup_file


def _run_build_ext(setup_file: Path, cwd: Path) -> None:
    """Run build_ext --inplace and clean up build artifacts."""
    run(
        [sys.executable, setup_file.name, "build_ext", "--inplace"],
        cwd=cwd,
    )
    setup_file.unlink()
    shutil.rmtree(cwd / "build", ignore_errors=True)


# ---------------------------------------------------------------------------
# Build stages
# ---------------------------------------------------------------------------

def validate_env(profile: str | None = None) -> dict[str, object]:
    """Check the build environment."""
    if profile == TAIMIDE_JETSON_PROFILE:
        return validate_taimide_jetson_env()

    arch = platform.machine()
    if arch != "aarch64":
        print(f"WARNING: expected aarch64, got {arch}. Wheel tag will reflect this platform.")

    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"Python {py} on {arch}")

    for pkg in ("Cython", "setuptools", "wheel", "numpy"):
        try:
            __import__(pkg)
        except ImportError:
            print(f"ERROR: {pkg} not installed. Run: pip install cython setuptools wheel numpy")
            sys.exit(1)

    return {
        "platform": platform.platform(),
        "architecture": arch,
        "python": platform.python_version(),
    }


def copy_to_staging(plan: CompilePlan) -> None:
    """Copy needed source files to the staging directory."""
    print("\n[1/6] Copying source files to staging directory...")
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir()

    def _copy(src_rel: str) -> None:
        src = PROJECT_ROOT / src_rel
        dst = BUILD_DIR / src_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    for src_rel in plan.keenchic_cython.values():
        _copy(src_rel)

    for src_rel in plan.keep_py:
        _copy(src_rel)

    for group in plan.dotted_groups:
        cwd = group["cwd"]
        for src_rel in group["extensions"].values():
            _copy(os.path.join(cwd, src_rel))

    for group in plan.bare_groups:
        cwd = group["cwd"]
        for src_rel in group["extensions"].values():
            _copy(os.path.join(cwd, src_rel))

    for weight_dir in plan.weight_dirs:
        src_dir = PROJECT_ROOT / weight_dir
        dst_dir = BUILD_DIR / weight_dir
        if src_dir.exists():
            shutil.copytree(src_dir, dst_dir)

    for weight_file in plan.weight_files:
        _copy(weight_file)

    for init_dir in plan.init_dirs:
        init_path = BUILD_DIR / init_dir / "__init__.py"
        init_path.parent.mkdir(parents=True, exist_ok=True)
        if not init_path.exists():
            init_path.write_text("")

    print(f"  Staging directory: {BUILD_DIR}")


def compile_keenchic_modules(plan: CompilePlan) -> None:
    """Compile keenchic.* modules with Cython (dotted module names)."""
    print("\n[2/6] Compiling keenchic.* modules...")
    setup_file = _write_compile_setup(BUILD_DIR, plan.keenchic_cython)
    _run_build_ext(setup_file, BUILD_DIR)


def compile_submodule_dotted(plan: CompilePlan) -> None:
    """Compile submodule files imported with dotted names."""
    if not plan.dotted_groups:
        print("\n[3/6] No dotted submodule modules to compile, skipping.")
        return

    print("\n[3/6] Compiling submodule dotted modules...")
    for group in plan.dotted_groups:
        cwd = BUILD_DIR / group["cwd"]
        print(f"  Directory: {group['cwd']}")
        setup_file = _write_compile_setup(cwd, group["extensions"])
        _run_build_ext(setup_file, cwd)


def compile_submodule_bare(plan: CompilePlan) -> None:
    """Compile submodule files imported with bare names."""
    if not plan.bare_groups:
        print("\n[4/6] No bare submodule modules to compile, skipping.")
        return

    print("\n[4/6] Compiling submodule bare modules...")
    for group in plan.bare_groups:
        cwd = BUILD_DIR / group["cwd"]
        print(f"  Directory: {group['cwd']}")
        setup_file = _write_compile_setup(cwd, group["extensions"])
        _run_build_ext(setup_file, cwd)


def cleanup_staging(plan: CompilePlan) -> None:
    """Remove .py source for compiled modules and build intermediates."""
    print("\n[5/6] Cleaning staging directory...")
    removed = 0

    for src_rel in plan.keenchic_cython.values():
        py_file = BUILD_DIR / src_rel
        if py_file.exists():
            py_file.unlink()
            removed += 1

    for group in plan.dotted_groups:
        cwd = group["cwd"]
        for src_rel in group["extensions"].values():
            py_file = BUILD_DIR / cwd / src_rel
            if py_file.exists():
                py_file.unlink()
                removed += 1

    for group in plan.bare_groups:
        cwd = group["cwd"]
        for src_rel in group["extensions"].values():
            py_file = BUILD_DIR / cwd / src_rel
            if py_file.exists():
                py_file.unlink()
                removed += 1

    for pattern in ("**/*.c", "**/*.cpp", "**/*.html"):
        for f in BUILD_DIR.glob(pattern):
            f.unlink()
            removed += 1

    print(f"  Removed {removed} files")


def _plan_input_paths(
    plan: CompilePlan,
    selected: dict[str, AlgoSpec],
) -> list[Path]:
    relative_paths: set[str] = {
        "build_wheel.py",
        *(str(spec.descriptor_path.relative_to(PROJECT_ROOT)) for spec in selected.values()),
        *plan.keenchic_cython.values(),
        *plan.keep_py,
        *plan.weight_files,
    }
    for group in plan.dotted_groups + plan.bare_groups:
        for source in group["extensions"].values():
            relative_paths.add(os.path.join(group["cwd"], source))
    return [PROJECT_ROOT / path for path in sorted(relative_paths)]


def _optional_git_revision(path: Path) -> str | None:
    environment = dict(os.environ)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def create_build_manifest(
    *,
    release_version: str,
    package_version: str,
    profile: str,
    edition: str,
    selected: dict[str, AlgoSpec],
    plan: CompilePlan,
    selected_engines: dict[str, SelectedEngine],
    runtime: dict[str, object],
) -> dict[str, object]:
    """Create traceable metadata for the actual inputs packaged by this build."""
    input_hashes = {
        str(path.relative_to(PROJECT_ROOT)): _sha256(path)
        for path in _plan_input_paths(plan, selected)
    }
    aggregate = hashlib.sha256()
    for path, digest in sorted(input_hashes.items()):
        aggregate.update(path.encode())
        aggregate.update(b"\0")
        aggregate.update(digest.encode())
        aggregate.update(b"\n")

    return {
        "schema_version": 1,
        "release_version": release_version,
        "package_version": package_version,
        "profile": profile,
        "edition": edition,
        "algorithms": sorted(selected),
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime,
        "build_tools": {
            name: _distribution_version(name)
            for name in ("Cython", "setuptools", "wheel", "numpy")
        },
        "engines": {
            name: {
                "path": str(engine.path.relative_to(PROJECT_ROOT)),
                "filename": engine.path.name,
                "version_date": engine.version_date,
                "sha256": engine.sha256,
            }
            for name, engine in sorted(selected_engines.items())
        },
        "source": {
            "git_revision": _optional_git_revision(PROJECT_ROOT),
            "ocr_submodule_revision": _optional_git_revision(
                PROJECT_ROOT / "keenchic/inspections/ocr"
            ),
            "aggregate_sha256": aggregate.hexdigest(),
            "inputs": input_hashes,
        },
    }


def write_manifest_to_staging(manifest: dict[str, object]) -> None:
    manifest_path = BUILD_DIR / "keenchic/build_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def create_target_constraints(install_requires: list[str]) -> str:
    """Pin installed direct dependencies without adding local/VCS references."""
    lines: list[str] = []
    for requirement in install_requires:
        match = re.match(r"[A-Za-z0-9_.-]+", requirement)
        if match is None:
            raise RuntimeError(f"cannot parse dependency name: {requirement}")
        name = match.group(0)
        version = _distribution_version(name)
        lines.append(f"{name}=={version}")
    return "\n".join(sorted(set(lines), key=str.lower)) + "\n"


def _version_tag(
    selected_names: list[str],
    all_names: set[str],
    edition: str = "standard",
    base_version: str = VERSION,
    profile: str | None = None,
) -> str:
    """Return version string; adds PEP 440 local tag for subset builds or non-standard editions."""
    if profile == TAIMIDE_JETSON_PROFILE:
        return f"{base_version}+taimide"

    parts: list[str] = []

    if edition != "standard":
        parts.append(edition)

    if set(selected_names) != all_names:
        def slugify(n: str) -> str:
            return n.replace("/", "_").replace("-", "_")
        parts.extend(slugify(n) for n in sorted(selected_names))

    if parts:
        return f"{base_version}+{'.' .join(parts)}"
    return base_version


def build_wheel(
    plan: CompilePlan,
    selected_names: list[str],
    all_names: set[str],
    edition: str = "standard",
    *,
    base_version: str = VERSION,
    profile: str | None = None,
    output_dir: Path = DIST_DIR,
) -> Path:
    """Generate setup.py for packaging and build the wheel."""
    print("\n[6/6] Building wheel...")

    version = _version_tag(
        selected_names,
        all_names,
        edition=edition,
        base_version=base_version,
        profile=profile,
    )

    packages = sorted(
        str(init.parent.relative_to(BUILD_DIR)).replace(os.sep, ".")
        for init in BUILD_DIR.glob("**/__init__.py")
    )

    package_data: dict[str, list[str]] = {}
    for pkg in packages:
        pkg_dir = BUILD_DIR / pkg.replace(".", os.sep)
        patterns = []
        if any(pkg_dir.glob("*.so")):
            patterns.append("*.so")
        if (pkg_dir / "weights").is_dir():
            patterns.append("weights/*")
        if pkg == "keenchic" and (pkg_dir / "build_manifest.json").is_file():
            patterns.append("build_manifest.json")
        if patterns:
            package_data[pkg] = patterns

    setup_content = textwrap.dedent(f"""\
        from setuptools import setup
        from setuptools.dist import Distribution as _Distribution


        class BinaryDistribution(_Distribution):
            \"\"\"Force platform-specific wheel (not pure-python).\"\"\"
            def has_ext_modules(self):
                return True


        setup(
            name="{BASE_PACKAGE_NAME}",
            version="{version}",
            python_requires={">=3.10,<3.11" if profile == TAIMIDE_JETSON_PROFILE else ">=3.10"!r},
            py_modules=["main", "serve"],
            packages={packages!r},
            package_data={package_data!r},
            install_requires={TAIMIDE_JETSON_INSTALL_REQUIRES if profile == TAIMIDE_JETSON_PROFILE else INSTALL_REQUIRES!r},
            entry_points={{
                "console_scripts": ["keenchic-serve=serve:main"],
            }},
            distclass=BinaryDistribution,
        )
    """)
    (BUILD_DIR / "setup.py").write_text(setup_content)

    (BUILD_DIR / "pyproject.toml").write_text(textwrap.dedent("""\
        [build-system]
        requires = ["setuptools", "wheel"]
        build-backend = "setuptools.build_meta"
    """))

    output_dir.mkdir(parents=True, exist_ok=True)
    before = set(output_dir.glob("keenchic_api_gateway-*.whl"))
    run(
        [
            sys.executable, "-m", "pip", "wheel",
            "--no-deps", "--no-build-isolation",
            "-w", str(output_dir), ".",
        ],
        cwd=BUILD_DIR,
    )
    new_wheels = set(output_dir.glob("keenchic_api_gateway-*.whl")) - before
    if len(new_wheels) != 1:
        raise RuntimeError(
            f"expected one new wheel in {output_dir}, found {len(new_wheels)}"
        )
    return new_wheels.pop()


def _write_text(path: Path, content: str) -> None:
    path.write_text(content)


def write_release_sidecars(
    output_dir: Path,
    wheel_path: Path,
    manifest: dict[str, object],
    constraints: str,
) -> tuple[Path, Path, Path]:
    manifest_path = output_dir / "build-manifest.json"
    constraints_path = output_dir / "target-constraints.txt"
    checksums_path = output_dir / "SHA256SUMS"

    _write_text(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _write_text(constraints_path, constraints)
    checksum_lines = [
        f"{_sha256(path)}  {path.name}"
        for path in (wheel_path, manifest_path, constraints_path)
    ]
    _write_text(checksums_path, "\n".join(checksum_lines) + "\n")
    return manifest_path, constraints_path, checksums_path


def validate_wheel_archive(
    wheel_path: Path,
    package_version: str,
    selected_engines: dict[str, SelectedEngine],
) -> None:
    """Verify profile-specific metadata and that only selected engines shipped."""
    with zipfile.ZipFile(wheel_path) as archive:
        members = archive.namelist()
        metadata_names = [name for name in members if name.endswith(".dist-info/METADATA")]
        wheel_names = [name for name in members if name.endswith(".dist-info/WHEEL")]
        if len(metadata_names) != 1 or len(wheel_names) != 1:
            raise RuntimeError("wheel must contain exactly one METADATA and WHEEL file")

        metadata = archive.read(metadata_names[0]).decode()
        wheel_metadata = archive.read(wheel_names[0]).decode()
        if f"Version: {package_version}\n" not in metadata:
            raise RuntimeError(f"wheel metadata version is not {package_version}")
        if not re.search(
            rf"^Requires-Dist: pycuda\s*(?:\(==|==)\s*{re.escape(TAIMIDE_JETSON_PYCUDA)}",
            metadata,
            flags=re.MULTILINE,
        ):
            raise RuntimeError("wheel metadata must require exact PyCUDA version")
        if "Tag: cp310-cp310-linux_aarch64" not in wheel_metadata:
            raise RuntimeError("wheel tag is not cp310-cp310-linux_aarch64")

        trt_members = sorted(Path(name).name for name in members if name.endswith(".trt"))
        expected_engines = sorted(engine.path.name for engine in selected_engines.values())
        if trt_members != expected_engines:
            raise RuntimeError(
                f"wheel engines differ: expected {expected_engines}, got {trt_members}"
            )
        manifest_names = [
            name
            for name in members
            if name == "keenchic/build_manifest.json"
            or name.endswith(".data/purelib/keenchic/build_manifest.json")
        ]
        if len(manifest_names) != 1:
            raise RuntimeError("wheel must contain exactly one embedded build manifest")
        embedded_manifest = json.loads(archive.read(manifest_names[0]))
        if embedded_manifest.get("package_version") != package_version:
            raise RuntimeError("embedded build manifest version does not match wheel")

        for module_name in TAIMIDE_JETSON_PROTECTED_MODULES:
            module_path = module_name.replace(".", "/")
            compiled_matches = [
                name
                for name in members
                if re.search(
                    rf"(?:^|/){re.escape(module_path)}\.cpython-310-.*\.so$",
                    name,
                )
            ]
            source_matches = [
                name
                for name in members
                if name == f"{module_path}.py"
                or name.endswith(f"/{module_path}.py")
            ]
            if len(compiled_matches) != 1 or source_matches:
                raise RuntimeError(
                    f"protected module {module_name} must ship as one .so and no .py"
                )

        bundled_gpu_bindings = [
            name
            for name in members
            if name.endswith(("libcuda.so", "libcuda.so.1"))
            or "/pycuda/" in name
            or "/tensorrt/" in name
        ]
        if bundled_gpu_bindings:
            raise RuntimeError(
                "wheel unexpectedly bundles GPU runtime files: "
                + ", ".join(bundled_gpu_bindings)
            )


def find_cached_pycuda_wheel() -> Path:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "cache", "dir"],
        check=True,
        capture_output=True,
        text=True,
    )
    cache_dir = Path(result.stdout.strip())
    wheel_name = (
        f"pycuda-{TAIMIDE_JETSON_PYCUDA}-cp310-cp310-linux_aarch64.whl"
    )
    matches = sorted((cache_dir / "wheels").rglob(wheel_name))
    if not matches:
        raise RuntimeError(
            f"cached {wheel_name} not found; prebuild PyCUDA on the Build Jetson"
        )
    return matches[-1]


def _installed_engine_verification_script(engine_paths: list[str]) -> str:
    return textwrap.dedent(
        f"""\
        import json
        import gc
        from pathlib import Path

        import keenchic
        import pycuda.driver as cuda
        import tensorrt as trt

        package_root = Path(keenchic.__file__).resolve().parent
        engine_paths = json.loads({json.dumps(json.dumps(engine_paths))})
        cuda.init()
        context = cuda.Device(0).make_context()
        loaded_engines = []
        runtime = None
        try:
            runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
            for relative_path in engine_paths:
                path = package_root / relative_path
                engine = runtime.deserialize_cuda_engine(path.read_bytes())
                if engine is None:
                    raise RuntimeError(f"could not deserialize {{path}}")
                loaded_engines.append(engine)
                engine = None
        finally:
            loaded_engines.clear()
            runtime = None
            gc.collect()
            context.pop()
            context.detach()
        print(package_root)
        """
    )


def verify_installed_wheel(
    wheel_path: Path,
    constraints_path: Path,
    selected_engines: dict[str, SelectedEngine],
) -> None:
    """Install and exercise the wheel outside the source repository."""
    cached_pycuda = find_cached_pycuda_wheel()
    with tempfile.TemporaryDirectory(
        prefix=".taimide-wheel-verify-",
        dir=PROJECT_ROOT,
    ) as temp_dir_value:
        temp_dir = Path(temp_dir_value)
        venv_dir = temp_dir / "venv"
        run([sys.executable, "-m", "venv", "--system-site-packages", str(venv_dir)])
        python_path = venv_dir / "bin/python"
        serve_path = venv_dir / "bin/keenchic-serve"

        run(
            [
                str(python_path),
                "-m",
                "pip",
                "install",
                "--only-binary=:all:",
                "--constraint",
                str(constraints_path),
                str(cached_pycuda),
            ],
            cwd=temp_dir,
        )
        run(
            [
                str(python_path),
                "-m",
                "pip",
                "install",
                "--only-binary=:all:",
                "--constraint",
                str(constraints_path),
                str(wheel_path),
            ],
            cwd=temp_dir,
        )
        run([str(serve_path), "--help"], cwd=temp_dir)

        engine_paths = [
            str(engine.path.relative_to(PROJECT_ROOT / "keenchic"))
            for engine in selected_engines.values()
        ]
        run(
            [
                str(python_path),
                "-B",
                "-c",
                _installed_engine_verification_script(engine_paths),
            ],
            cwd=temp_dir,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build keenchic-API-Gateway wheel (descriptor-driven, Cython + weights).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              python3 build_wheel.py --profile taimide-jetson
              python3 build_wheel.py --profile taimide-jetson \\
                                     --release-version 2026.8.23
              python3 build_wheel.py                          # all algorithms
              python3 build_wheel.py --list                   # list available
              python3 build_wheel.py -a ocr/datecode-num      # single algorithm
              python3 build_wheel.py -a ocr/datecode-num \\
                                     -a ocr/pill-count        # subset
        """),
    )
    p.add_argument(
        "-a", "--algorithm",
        action="append",
        default=[],
        metavar="NAME",
        help="Inspection name to include (repeatable). Default: all discovered.",
    )
    p.add_argument(
        "--edition",
        choices=["standard", "taimide"],
        default=None,
        help="Build edition: standard (default) or taimide (includes taimide-specific modules).",
    )
    p.add_argument(
        "--profile",
        choices=[TAIMIDE_JETSON_PROFILE],
        help="Build a strict deployment profile.",
    )
    p.add_argument(
        "--release-version",
        help=(
            "Calendar release version YYYY.M.D[.N]. With --profile, defaults "
            "to the system date and next same-day revision."
        ),
    )
    p.add_argument(
        "--engine-date",
        action="append",
        default=[],
        metavar="NAME=YYYYMMDD",
        help="Override a profile engine date (repeatable).",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List discovered algorithms and exit.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    specs = discover_descriptors()

    if args.list:
        print("Available algorithms:")
        for name in sorted(specs):
            print(f"  {name}")
        return

    profile = args.profile
    if profile == TAIMIDE_JETSON_PROFILE:
        if args.edition is not None or args.algorithm:
            raise SystemExit(
                "ERROR: --profile taimide-jetson cannot be combined with "
                "--edition or --algorithm"
            )
        try:
            release_version = resolve_release_version(args.release_version)
            engine_overrides = parse_engine_date_overrides(args.engine_date)
        except ValueError as exc:
            raise SystemExit(f"ERROR: {exc}") from exc
        if args.release_version is None:
            print(f"Auto-selected release version: {release_version}")
        edition = "taimide"
        selected = select_algorithms(specs, list(TAIMIDE_JETSON_ALGORITHMS))
        output_dir = DIST_DIR / release_version
        if output_dir.exists():
            raise SystemExit(f"ERROR: release output already exists: {output_dir}")
    else:
        if args.release_version or args.engine_date:
            raise SystemExit(
                "ERROR: --release-version and --engine-date require --profile"
            )
        release_version = VERSION
        engine_overrides = {}
        edition = args.edition or "standard"
        selected = select_algorithms(specs, args.algorithm)
        output_dir = DIST_DIR

    runtime = validate_env(profile)
    if profile == TAIMIDE_JETSON_PROFILE:
        try:
            selected_engines = select_profile_engines(
                selected,
                profile,
                engine_overrides,
            )
            validate_selected_engines(selected_engines)
        except (RuntimeError, ValueError) as exc:
            raise SystemExit(f"ERROR: {exc}") from exc
    else:
        selected_engines = {}

    package_version = _version_tag(
        list(selected),
        set(specs),
        edition=edition,
        base_version=release_version,
        profile=profile,
    )
    print(
        f"\nBuilding {BASE_PACKAGE_NAME} v{package_version} "
        f"(edition: {edition}, profile: {profile or 'none'})"
    )
    print(f"Algorithms ({len(selected)}): {', '.join(sorted(selected))}")
    if selected_engines:
        print(
            "Engines: "
            + ", ".join(
                f"{name}={engine.path.name}"
                for name, engine in sorted(selected_engines.items())
            )
        )
    print("=" * 60)

    plan = compile_plan(
        selected,
        edition=edition,
        selected_engines=selected_engines if profile else None,
    )
    copy_to_staging(plan)
    if profile == TAIMIDE_JETSON_PROFILE:
        manifest = create_build_manifest(
            release_version=release_version,
            package_version=package_version,
            profile=profile,
            edition=edition,
            selected=selected,
            plan=plan,
            selected_engines=selected_engines,
            runtime=runtime,
        )
        write_manifest_to_staging(manifest)
        constraints = create_target_constraints(TAIMIDE_JETSON_INSTALL_REQUIRES)
    else:
        manifest = None
        constraints = ""

    compile_keenchic_modules(plan)
    compile_submodule_dotted(plan)
    compile_submodule_bare(plan)
    cleanup_staging(plan)
    wheel_path = build_wheel(
        plan,
        list(selected),
        set(specs),
        edition=edition,
        base_version=release_version,
        profile=profile,
        output_dir=output_dir,
    )

    shutil.rmtree(BUILD_DIR)

    if profile == TAIMIDE_JETSON_PROFILE:
        assert manifest is not None
        manifest_path, constraints_path, checksums_path = write_release_sidecars(
            output_dir,
            wheel_path,
            manifest,
            constraints,
        )
        validate_wheel_archive(wheel_path, package_version, selected_engines)
        verify_installed_wheel(wheel_path, constraints_path, selected_engines)
        print(f"Manifest: {manifest_path}")
        print(f"Constraints: {constraints_path}")
        print(f"Checksums: {checksums_path}")

    print("\n" + "=" * 60)
    size_mb = wheel_path.stat().st_size / 1024 / 1024
    print(f"Wheel built and verified: {wheel_path.name}")
    print(f"Size: {size_mb:.1f} MB")
    print(f"Location: {wheel_path}")


if __name__ == "__main__":
    main()
