#!/usr/bin/env python3
"""
TAVS-ESP Test Runner

Runs all TAVS-ESP test suites from the src/tests directory.
Usage: python run_tests.py [test_name]
"""

import sys
import subprocess
from pathlib import Path


def run_test(test_name: str) -> bool:
    """Run a specific test module."""
    try:
        result = subprocess.run([
            sys.executable, "-m", f"src.tests.{test_name}"
        ], capture_output=True, text=True, cwd=Path.cwd())

        print(f"=== {test_name} ===")
        print(result.stdout)

        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        if result.returncode != 0:
            print(f"❌ {test_name} FAILED (exit code: {result.returncode})")
            return False
        else:
            print(f"✅ {test_name} PASSED")
            return True

    except Exception as e:
        print(f"❌ {test_name} ERROR: {e}")
        return False


def main():
    """Run TAVS-ESP test suite."""
    print("🧪 TAVS-ESP Complete Test Suite")
    print("=" * 50)

    # Available tests
    all_tests = [
        "test_csprng",
        "test_tavs_scheduler",
        "test_integration",
        "test_tavs_esp_strategy",
        "test_tavs_flower_client",
        "test_end_to_end_pipeline"
    ]

    # Check if specific test requested
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        if test_name.startswith("test_"):
            test_name = test_name[5:]  # Remove 'test_' prefix if provided

        target_test = f"test_{test_name}"
        if target_test in all_tests:
            success = run_test(target_test)
            sys.exit(0 if success else 1)
        else:
            print(f"❌ Unknown test: {test_name}")
            print(f"Available tests: {', '.join([t[5:] for t in all_tests])}")
            sys.exit(1)

    # Run all tests
    results = []
    for test in all_tests:
        success = run_test(test)
        results.append((test, success))
        print()  # Add spacing between tests

    # Summary
    passed = sum(1 for _, success in results if success)
    total = len(results)

    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} PASSED")

    for test, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test}: {status}")

    if passed == total:
        print("\n🎯 All TAVS-ESP tests PASSED!")
        sys.exit(0)
    else:
        print(f"\n❌ {total - passed} tests FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()