#!/usr/bin/env python3
"""Every test_* function in a suite whose main() aggregates with all([...])
must actually RETURN a truthy value.

WHY THIS EXISTS
test_end_to_end_pipeline.py's main() does
    if all([success1, ..., success8]):
and test_server_strategy_creation had no return statement at all, so it
returned None. The suite reported "Some End-to-End Pipeline tests FAILED"
while every individual check printed a tick, and the nearest visible message
was an unrelated warning from a later test. That combination is very
expensive to diagnose: the failure is invisible in the output and the
loudest line in the log is a red herring.

This is a static check -- it parses the files rather than running them, so
it costs milliseconds and does not need torch, a dataset or a GPU.
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUITES = ["test_end_to_end_pipeline.py", "test_tavs_esp_strategy.py"]


_NESTED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def _own_returns(node):
    """Return statements belonging to THIS function, not to nested ones.

    ast.walk descends into nested defs and classes, so a mock helper with a
    bare `return` inside the test made the check fire on the test itself.
    That false positive is worse than no check: a guard nobody trusts gets
    deleted.
    """
    out = []
    stack = list(node.body)
    while stack:
        n = stack.pop()
        if isinstance(n, _NESTED):
            continue                      # a different function's returns
        if isinstance(n, ast.Return):
            out.append(n)
        stack.extend(ast.iter_child_nodes(n))
    return out


def _always_returns(body):
    """Does every path through this statement list end in return/raise?

    Needed because a function whose last statement is `with ...:` or
    `try: ... except: ...` does NOT fall off the end if every branch
    returns. Treating the last statement as a plain node flagged a
    correct test.
    """
    if not body:
        return False
    last = body[-1]
    if isinstance(last, (ast.Return, ast.Raise)):
        return True
    if isinstance(last, ast.With):
        return _always_returns(last.body)
    if isinstance(last, ast.If):
        return bool(last.orelse) and _always_returns(last.body) \
            and _always_returns(last.orelse)
    if isinstance(last, ast.Try):
        if last.finalbody and _always_returns(last.finalbody):
            return True
        body_ok = _always_returns(last.body) if not last.orelse \
            else _always_returns(last.orelse)
        return body_ok and bool(last.handlers) and all(
            _always_returns(h.body) for h in last.handlers)
    return False


def offenders(path: Path):
    """test_* functions that can return None: a bare return, or a path that
    falls off the end."""
    tree = ast.parse(path.read_text())
    bad = []
    for node in tree.body:
        if not (isinstance(node, ast.FunctionDef) and node.name.startswith("test_")):
            continue
        if any(r.value is None for r in _own_returns(node)):
            bad.append((node.name, "has a bare `return`"))
        elif not _always_returns(node.body):
            bad.append((node.name, "falls off the end without returning"))
    return bad


def test_all_suite_tests_return_truthy():
    failures = []
    for name in SUITES:
        p = ROOT / "src" / "tests" / name
        if not p.exists():
            continue
        for fn, why in offenders(p):
            failures.append(f"{name}::{fn} {why}")
    assert not failures, (
        "these test functions return None, so an all([...]) aggregator in "
        "main() silently reports the whole suite as FAILED while every "
        "printed check passes:\n  " + "\n  ".join(failures)
    )
    print(f"✓ every test_* in {len(SUITES)} aggregated suites returns a value")
    return True


def test_detector_catches_a_missing_return():
    """The check must actually fire. Without this, a bug in offenders()
    would make the guard vacuous -- the same failure mode it guards against."""
    import tempfile
    src = (
        "def test_ok():\n    print('x')\n    return True\n\n"
        "def test_falls_off():\n    print('x')\n\n"
        "def test_bare_return():\n    return\n\n"
        "def test_raises():\n    raise SystemExit(1)\n\n"
        # Both of these were FALSE POSITIVES in the first version of the
        # detector and are the reason _own_returns and _always_returns
        # exist. A mock helper's bare return belongs to the helper, and a
        # with/try whose every branch returns does not fall off the end.
        "def test_nested_bare_return_is_not_mine():\n"
        "    class Mock:\n        def f(self):\n            return\n"
        "    return True\n\n"
        "def test_with_try_all_paths_return():\n"
        "    with open('/dev/null') as f:\n"
        "        try:\n            return True\n"
        "        except Exception:\n            return True\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        tmp = Path(f.name)
    try:
        names = {n for n, _ in offenders(tmp)}
        assert names == {"test_falls_off", "test_bare_return"}, (
            f"expected exactly the two real offenders, got {names}")
    finally:
        tmp.unlink()
    print("✓ the detector flags a missing and a bare return, and nothing else")
    return True


def main():
    ok = [test_detector_catches_a_missing_return(),
          test_all_suite_tests_return_truthy()]
    if all(ok):
        print("\n🎯 All suite-return-contract tests PASSED!")
        return True
    return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
