# Auto-start coverage in every Python process that has COVERAGE_PROCESS_START set.
# Placed on PYTHONPATH (=/workspace/verl) by entrypoint.sh so it is imported at
# interpreter startup, capturing coverage for pytest and any ray worker/actor
# subprocesses.
import os
import sys

try:
    import coverage

    coverage.process_startup()
except Exception as exc:  # pragma: no cover
    # Only warn when coverage was actually requested for this process, so a
    # missing coverage install doesn't spam unrelated Python processes -- but a
    # real failure to start (which would silently drop coverage data) is visible.
    if os.environ.get("COVERAGE_PROCESS_START"):
        print(
            "WARNING: sitecustomize could not start coverage (%r); "
            "subprocess coverage data may be missing." % (exc,),
            file=sys.stderr,
        )


def _chain_to_shadowed_sitecustomize() -> None:
    """Run any sitecustomize this file shadowed.

    Python imports only the FIRST ``sitecustomize`` on ``sys.path``. Ours is
    prepended via PYTHONPATH, so a base-image sitecustomize (which may perform
    required runtime setup, e.g. ROCm library preloading) would silently stop
    running. Locate and execute the next one instead of replacing it.
    """
    import importlib.util
    from importlib.machinery import PathFinder

    here = os.path.dirname(os.path.abspath(__file__))
    others = [p for p in sys.path if p and os.path.abspath(p) != here]

    spec = PathFinder.find_spec("sitecustomize", others) if others else None
    if spec is None or spec.origin is None:
        return
    if os.path.abspath(spec.origin) == os.path.abspath(__file__):
        return

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


try:
    _chain_to_shadowed_sitecustomize()
except Exception as exc:  # pragma: no cover
    print(
        "WARNING: sitecustomize could not chain to the shadowed "
        "sitecustomize (%r)." % (exc,),
        file=sys.stderr,
    )
