from pathlib import Path

import fastchrf


def test_wheel_contains_type_information():
    package_directory = Path(fastchrf.__file__).parent

    assert (package_directory / "__init__.pyi").is_file()
    assert (package_directory / "py.typed").is_file()
