"""Compatibility probe for the installed cayleypy version.

The fast engine and the in-place patch machinery rely on cayleypy *internals*
(alpha-stage API, not yet frozen). ``run_probe`` asserts importability and rough
signatures of every internal ``cayleypy_fast`` touches. Any mismatch makes
``enable()`` a warning + no-op, so behaviour falls back to legacy cayleypy.
"""

import dataclasses
import importlib
import inspect
from dataclasses import dataclass, field
from typing import Optional

# Module-level helpers used by the forked mode methods (plan section "Packaging & activation").
_REQUIRED_MODULE_FUNCTIONS = {
    "cayleypy.torch_utils": ["isin_via_searchsorted"],
    "cayleypy.algo.beam_search": [
        "_check_path_found",
        "_restore_path",
        "_init_predictor",
        "_encode_and_dedupe_start",
        "_setup_path_device_and_restore",
        "_precompute_mitm",
        "_early_return_if_at_dest",
        "_finalize_not_found",
        "_cuda_sync",
    ],
}

# Classes (and the methods on them) the engine/patch rely on.
_REQUIRED_CLASS_METHODS = {
    "cayleypy.torch_utils": {
        "TorchHashSet": ["add_sorted_hashes", "get_mask_to_remove_seen_hashes", "get_merged_sorted"],
    },
    "cayleypy.hasher": {
        # Method set pins the hasher variant the engine derives its permuted hash
        # vectors from (dual-int32 CPU path + int64 GPU path + splitmix64).
        "StateHasher": ["_make_hashes_dual_int32", "_make_hashes_cpu_and_modern_gpu", "_hash_splitmix64"],
    },
    "cayleypy.algo.beam_search": {
        "BeamSearchAlgorithm": [
            "search",
            "search_simple",
            "search_advanced",
            "search_iterated",
            "search_iterated_batched",
        ],
        "BeamSearchResult": [],
        "_BeamSearchProfile": ["reset_step", "format_line"],
    },
}

# ``_restore_path`` must take ``destination_state`` (post-audit-fix cayleypy; the
# pre-fix version silently restored layer-0 paths to central_state — probe must
# reject that variant, not crash at restore time).
_EXPECTED_RESTORE_PATH_PARAMS = {"found_layer_id", "restore_path_hashes", "destination_state"}

# Keyword parameters of ``BeamSearchAlgorithm.search`` that the patch forwards (rough signature check).
_EXPECTED_SEARCH_PARAMS = {
    "start_state",
    "destination_state",
    "beam_mode",
    "predictor",
    "beam_width",
    "max_steps",
    "history_depth",
    "return_path",
    "path_device",
    "hashed_neigbourhood",
    "memory_cleanup",
    "verbose",
}

# Expected ``BeamSearchResult`` dataclass fields.
_EXPECTED_RESULT_FIELDS = {"path_found", "path_length", "path", "debug_scores", "graph"}

# Modules whose ``BeamSearchAlgorithm`` attribute is replaced by ``enable()``
# (covers both ``graph.beam_search`` and direct importers of the symbol).
PATCH_POINTS = ["cayleypy.cayley_graph", "cayleypy.algo", "cayleypy.algo.beam_search"]


@dataclass
class ProbeResult:
    """Outcome of the compatibility probe: ``ok`` flag plus human-readable problems."""

    ok: bool
    problems: list = field(default_factory=list)


def _import_module(name: str, problems: list) -> Optional[object]:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        problems.append(f"cannot import {name}: {exc}")
        return None


def _check_module_functions(modules: dict, problems: list) -> None:
    for module_name, func_names in _REQUIRED_MODULE_FUNCTIONS.items():
        module = modules.get(module_name)
        if module is None:
            continue
        for func_name in func_names:
            if not callable(getattr(module, func_name, None)):
                problems.append(f"{module_name}.{func_name} missing or not callable.")


def _check_class_methods(modules: dict, problems: list) -> None:
    for module_name, classes in _REQUIRED_CLASS_METHODS.items():
        module = modules.get(module_name)
        if module is None:
            continue
        for class_name, method_names in classes.items():
            cls = getattr(module, class_name, None)
            if not isinstance(cls, type):
                problems.append(f"{module_name}.{class_name} missing or not a class.")
                continue
            for method_name in method_names:
                if not callable(getattr(cls, method_name, None)):
                    problems.append(f"{module_name}.{class_name}.{method_name} missing or not callable.")


def _check_search_signature(beam_search_module: Optional[object], problems: list) -> None:
    if beam_search_module is None:
        return
    cls = getattr(beam_search_module, "BeamSearchAlgorithm", None)
    if cls is None or not callable(getattr(cls, "search", None)):
        return  # Already reported by _check_class_methods.
    params = set(inspect.signature(cls.search).parameters)
    missing = sorted(_EXPECTED_SEARCH_PARAMS - params)
    if missing:
        problems.append(f"BeamSearchAlgorithm.search signature drifted; missing params: {missing}.")


def _check_restore_path_signature(beam_search_module: Optional[object], problems: list) -> None:
    if beam_search_module is None:
        return
    func = getattr(beam_search_module, "_restore_path", None)
    if not callable(func):
        return  # Already reported by _check_module_functions.
    params = set(inspect.signature(func).parameters)
    missing = sorted(_EXPECTED_RESTORE_PATH_PARAMS - params)
    if missing:
        problems.append(f"_restore_path signature drifted; missing params: {missing}.")


def _check_result_fields(result_module: Optional[object], problems: list) -> None:
    if result_module is None:
        return
    cls = getattr(result_module, "BeamSearchResult", None)
    if cls is None:
        problems.append("cayleypy.algo.beam_search_result.BeamSearchResult missing.")
        return
    if not dataclasses.is_dataclass(cls):
        problems.append("BeamSearchResult is no longer a dataclass.")
        return
    field_names = {f.name for f in dataclasses.fields(cls)}
    missing = sorted(_EXPECTED_RESULT_FIELDS - field_names)
    if missing:
        problems.append(f"BeamSearchResult fields drifted; missing: {missing}.")


def _check_patch_points(modules: dict, problems: list) -> None:
    for module_name in PATCH_POINTS:
        module = modules.get(module_name)
        if module is None:
            continue
        if not isinstance(getattr(module, "BeamSearchAlgorithm", None), type):
            problems.append(f"patch point {module_name}.BeamSearchAlgorithm missing or not a class.")
    cg_module = modules.get("cayleypy.cayley_graph")
    bs_module = modules.get("cayleypy.algo.beam_search")
    if cg_module is None or bs_module is None:
        return
    cg_bsa = getattr(cg_module, "BeamSearchAlgorithm", None)
    bs_bsa = getattr(bs_module, "BeamSearchAlgorithm", None)
    if cg_bsa is None or bs_bsa is None:
        return
    # Skip the identity check when our own patched class is currently installed
    # (run_probe called after enable() must not report a false drift).
    if str(getattr(cg_bsa, "__module__", "")).startswith("cayleypy_fast"):
        return
    if cg_bsa is not bs_bsa:
        problems.append(
            "patch point drift: cayleypy.cayley_graph.BeamSearchAlgorithm no longer aliases "
            "cayleypy.algo.beam_search.BeamSearchAlgorithm."
        )


def run_probe() -> ProbeResult:
    """Assert that the installed cayleypy exposes every internal ``cayleypy_fast`` relies on."""
    problems: list = []
    module_names = set(_REQUIRED_MODULE_FUNCTIONS) | set(_REQUIRED_CLASS_METHODS) | set(PATCH_POINTS)
    module_names.add("cayleypy.algo.beam_search_result")
    modules = {name: _import_module(name, problems) for name in module_names}

    _check_module_functions(modules, problems)
    _check_class_methods(modules, problems)
    _check_search_signature(modules.get("cayleypy.algo.beam_search"), problems)
    _check_restore_path_signature(modules.get("cayleypy.algo.beam_search"), problems)
    _check_result_fields(modules.get("cayleypy.algo.beam_search_result"), problems)
    _check_patch_points(modules, problems)
    return ProbeResult(ok=not problems, problems=problems)
