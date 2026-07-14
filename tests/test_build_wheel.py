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
