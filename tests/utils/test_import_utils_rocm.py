# Copyright 2026 AMD
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ROCm code-coverage tests for verl.utils.import_utils.

This module is pure-python (no GPU / no heavy deps), so it runs anywhere the
verl package imports. It complements tests/utils/test_import_utils_on_cpu.py
(which only exercises load_extern_object) by covering the availability probes,
load_module (pkg:// + file:// + module_name caching + error paths), the
`deprecated` decorator (function and class forms), load_class_from_fqn, and the
deprecated load_extern_type wrapper.
"""

import os
import sys
import warnings

import pytest

from verl.utils.import_utils import (
    deprecated,
    import_external_libs,
    is_megatron_core_available,
    is_msprobe_available,
    is_nvtx_available,
    is_sglang_available,
    is_trl_available,
    is_vllm_available,
    load_class_from_fqn,
    load_extern_type,
    load_module,
)

TEST_MODULE_PATH = os.path.join(os.path.dirname(__file__), "_test_module.py")


def test_availability_probes_return_bool():
    # We only assert the type; the truth value depends on the environment.
    for probe in (
        is_megatron_core_available,
        is_vllm_available,
        is_sglang_available,
        is_nvtx_available,
        is_trl_available,
        is_msprobe_available,
    ):
        assert isinstance(probe(), bool)


def test_import_external_libs_variants():
    # None -> no-op
    assert import_external_libs(None) is None
    # single string is wrapped into a list
    import_external_libs("os")
    # list of importable modules
    import_external_libs(["os", "sys"])


def test_load_module_empty_returns_none():
    assert load_module("") is None


def test_load_module_pkg_prefix():
    mod = load_module("pkg://verl.utils.import_utils")
    assert mod is not None and hasattr(mod, "load_module")


def test_load_module_file_prefix_and_caching():
    name = "verl_cc_loaded_test_module"
    sys.modules.pop(name, None)
    try:
        mod = load_module(f"file://{TEST_MODULE_PATH}", module_name=name)
        assert mod is not None and hasattr(mod, "TestClass")
        # module_name was registered in sys.modules
        assert sys.modules.get(name) is mod
    finally:
        sys.modules.pop(name, None)


def test_load_module_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_module("/nonexistent/verl_cc/module.py")


def test_load_module_name_collision_raises():
    name = "verl_cc_collision_module"
    sys.modules[name] = object()  # a different object than what we will load
    try:
        with pytest.raises(RuntimeError):
            load_module(TEST_MODULE_PATH, module_name=name)
    finally:
        sys.modules.pop(name, None)


def test_deprecated_function_warns_and_calls():
    @deprecated(replacement="new_func")
    def old_func(a, b):
        return a + b

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert old_func(2, 3) == 5
    assert any(issubclass(w.category, FutureWarning) for w in caught)


def test_deprecated_function_without_replacement():
    @deprecated()
    def old_func():
        return "ok"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert old_func() == "ok"
    assert any(issubclass(w.category, FutureWarning) for w in caught)


def test_deprecated_class_warns_on_init():
    @deprecated(replacement="NewClass")
    class OldClass:
        def __init__(self, value):
            self.value = value

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        obj = OldClass(42)
    assert obj.value == 42
    assert any(issubclass(w.category, FutureWarning) for w in caught)


def test_load_class_from_fqn_success():
    cls = load_class_from_fqn("collections.OrderedDict")
    from collections import OrderedDict

    assert cls is OrderedDict


def test_load_class_from_fqn_no_dot_raises():
    with pytest.raises(ValueError):
        load_class_from_fqn("NoDotHere")


def test_load_class_from_fqn_bad_module_raises():
    with pytest.raises(ImportError):
        load_class_from_fqn("verl_cc_no_such_module.SomeClass")


def test_load_class_from_fqn_missing_attr_raises():
    with pytest.raises(AttributeError):
        load_class_from_fqn("collections.NoSuchClass")


def test_load_extern_type_deprecated_wrapper():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        TestClass = load_extern_type(TEST_MODULE_PATH, "TestClass")
    assert TestClass is not None and TestClass.__name__ == "TestClass"
    assert any(issubclass(w.category, FutureWarning) for w in caught)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
