"""
Code Execution Sandbox.
Safely executes user Python code with strict timeouts and import restrictions.
NOT a production-grade sandbox — for production, use Docker or Judge0.
"""

import subprocess
import tempfile
import os
import json
import time
import re
from backend.config import settings


BLOCKED_IMPORTS = settings.blocked_imports

IMPORT_CHECK_PATTERN = re.compile(
    r'(?:^|\n)\s*(?:import|from)\s+(' + '|'.join(re.escape(m) for m in BLOCKED_IMPORTS) + r')\b',
    re.MULTILINE
)

DANGEROUS_PATTERNS = [
    r'exec\s*\(',
    r'eval\s*\(',
    r'compile\s*\(',
    r'__import__\s*\(',
    r'open\s*\(',
    r'globals\s*\(\s*\)',
    r'locals\s*\(\s*\)',
]

DANGEROUS_RE = re.compile('|'.join(DANGEROUS_PATTERNS))


def check_code_safety(code: str) -> tuple[bool, str]:
    """
    Check if the code contains prohibited imports or dangerous patterns.

    Returns:
        (is_safe, reason) tuple
    """
    # Check blocked imports
    match = IMPORT_CHECK_PATTERN.search(code)
    if match:
        return False, f"Prohibited import detected: '{match.group(1)}'. For security, imports like os, sys, subprocess are not allowed."

    # Check dangerous patterns
    danger_match = DANGEROUS_RE.search(code)
    if danger_match:
        return False, f"Potentially dangerous code pattern detected: '{danger_match.group()}'. Please use only standard safe operations."

    return True, "Code passed safety checks"


def execute_code(code: str, test_cases: list = None) -> dict:
    """
    Execute Python code in an isolated subprocess.

    Args:
        code: User's Python code
        test_cases: Optional list of test cases to run

    Returns:
        {
            "success": bool,
            "stdout": str,
            "stderr": str,
            "passed_tests": int,
            "total_tests": int,
            "test_results": [...],
            "execution_time_ms": float,
            "safety_error": str or None
        }
    """
    # Safety check first
    is_safe, reason = check_code_safety(code)
    if not is_safe:
        return {
            "success": False,
            "stdout": "",
            "stderr": reason,
            "passed_tests": 0,
            "total_tests": len(test_cases) if test_cases else 0,
            "test_results": [],
            "execution_time_ms": 0,
            "safety_error": reason
        }

    # Build the execution script
    if test_cases:
        test_script = _build_test_script(code, test_cases)
    else:
        test_script = code

    # Execute in subprocess
    start_time = time.time()

    try:
        with tempfile.TemporaryDirectory(dir=tempfile.gettempdir()) as sandbox_dir:
            temp_path = os.path.join(sandbox_dir, "submission.py")
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(test_script)

            result = subprocess.run(
                ['python3', '-I', temp_path],
                capture_output=True,
                text=True,
                cwd=sandbox_dir,
                timeout=settings.code_execution_timeout,
                env={
                    'PATH': os.environ.get('PATH', '/usr/bin'),
                    'HOME': '/tmp',
                    'PYTHONDONTWRITEBYTECODE': '1',
                    'PYTHONNOUSERSITE': '1',
                }
            )

        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        stdout = result.stdout[:settings.max_output_length]
        stderr = result.stderr[:settings.max_output_length]

        # Parse test results if test cases were provided
        passed = 0
        total = len(test_cases) if test_cases else 0
        test_results = []

        if test_cases and stdout:
            try:
                # Look for our JSON test results marker
                lines = stdout.strip().split('\n')
                for line in lines:
                    if line.startswith('__TEST_RESULT__:'):
                        result_json = json.loads(line[16:])
                        test_results = result_json.get("results", [])
                        passed = sum(1 for r in test_results if r.get("passed"))
                        total = len(test_results)
                        # Remove test result lines from stdout
                        stdout = '\n'.join(l for l in lines if not l.startswith('__TEST_RESULT__:'))
                        break
            except (json.JSONDecodeError, IndexError):
                pass

        return {
            "success": result.returncode == 0,
            "stdout": stdout,
            "stderr": stderr,
            "passed_tests": passed,
            "total_tests": total,
            "test_results": test_results,
            "execution_time_ms": elapsed_ms,
            "safety_error": None
        }

    except subprocess.TimeoutExpired:
        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Execution timed out after {settings.code_execution_timeout} seconds. Check for infinite loops.",
            "passed_tests": 0,
            "total_tests": len(test_cases) if test_cases else 0,
            "test_results": [],
            "execution_time_ms": elapsed_ms,
            "safety_error": None
        }

    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Execution error: {str(e)}",
            "passed_tests": 0,
            "total_tests": len(test_cases) if test_cases else 0,
            "test_results": [],
            "execution_time_ms": 0,
            "safety_error": str(e)
        }


def _build_test_script(user_code: str, test_cases: list) -> str:
    """
    Build a test runner script that executes user code against test cases.
    """
    test_entries = json.dumps(test_cases)

    script = (
        "import ast\n"
        "import json\n"
        "import traceback\n\n"
        "# ---- User Code ----\n"
        f"{user_code}\n"
        "# ---- End User Code ----\n\n"
        "# ---- Test Runner ----\n"
        f"_test_cases = {test_entries}\n"
        "_results = []\n\n"
        "def _safe_parse(_value):\n"
        "    if isinstance(_value, (int, float, bool, list, tuple, dict)) or _value is None:\n"
        "        return _value\n"
        "    if not isinstance(_value, str):\n"
        "        return _value\n"
        "    try:\n"
        "        return ast.literal_eval(_value)\n"
        "    except Exception:\n"
        "        return _value\n\n"
        "def _find_solution_callable():\n"
        "    for _name in ('solution', 'solve', 'answer'):\n"
        "        _obj = globals().get(_name)\n"
        "        if callable(_obj):\n"
        "            return _obj\n"
        "    for _name, _obj in globals().items():\n"
        "        if _name.startswith('_'):\n"
        "            continue\n"
        "        if callable(_obj) and getattr(_obj, '__module__', '') == '__main__':\n"
        "            return _obj\n"
        "    return None\n\n"
        "def _parse_call_expression(_expr):\n"
        "    _node = ast.parse(_expr, mode='eval').body\n"
        "    if not isinstance(_node, ast.Call) or not isinstance(_node.func, ast.Name):\n"
        "        raise ValueError('Unsupported call expression format')\n"
        "    _name = _node.func.id\n"
        "    _args = [ast.literal_eval(arg) for arg in _node.args]\n"
        "    _kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in _node.keywords}\n"
        "    return _name, _args, _kwargs\n\n"
        "def _run_lru_case(_case_expr):\n"
        "    _cache_cls = globals().get('LRUCache')\n"
        "    if _cache_cls is None:\n"
        "        raise RuntimeError('LRUCache class not found')\n"
        "    _commands = [c.strip() for c in _case_expr.split(';') if c.strip()]\n"
        "    if not _commands:\n"
        "        raise RuntimeError('Empty LRUCache test case')\n"
        "    _ctor_name, _ctor_args, _ctor_kwargs = _parse_call_expression(_commands[0])\n"
        "    if _ctor_name != 'LRUCache':\n"
        "        raise RuntimeError('First command must construct LRUCache')\n"
        "    _cache = _cache_cls(*_ctor_args, **_ctor_kwargs)\n"
        "    _last_result = None\n"
        "    for _command in _commands[1:]:\n"
        "        _method_name, _args, _kwargs = _parse_call_expression(_command)\n"
        "        if not hasattr(_cache, _method_name):\n"
        "            raise AttributeError(f'Unknown method on LRUCache: {_method_name}')\n"
        "        _last_result = getattr(_cache, _method_name)(*_args, **_kwargs)\n"
        "    return _last_result\n\n"
        "for i, tc in enumerate(_test_cases):\n"
        "    try:\n"
        "        _input_raw = tc.get('input', '')\n"
        "        _expected_raw = tc.get('expected_output', '')\n"
        "        _expected_value = _safe_parse(_expected_raw)\n\n"
        "        if isinstance(_input_raw, str) and 'LRUCache(' in _input_raw and 'LRUCache' in globals():\n"
        "            _actual = _run_lru_case(_input_raw)\n"
        "        else:\n"
        "            _func = _find_solution_callable()\n"
        "            if _func is None:\n"
        "                raise RuntimeError('No callable solution found. Define solution(...) or solve(...).')\n"
        "            _input_value = _safe_parse(_input_raw)\n"
        "            if isinstance(_input_value, tuple):\n"
        "                _actual = _func(*_input_value)\n"
        "            else:\n"
        "                _actual = _func(_input_value)\n\n"
        "        _passed = (_actual == _expected_value) or (str(_actual).strip() == str(_expected_raw).strip())\n"
        "        _results.append({\n"
        "            'test': i + 1,\n"
        "            'passed': _passed,\n"
        "            'input': str(_input_raw),\n"
        "            'expected': str(_expected_raw),\n"
        "            'actual': str(_actual),\n"
        "            'description': tc.get('description', '')\n"
        "        })\n"
        "    except Exception:\n"
        "        _results.append({\n"
        "            'test': i + 1,\n"
        "            'passed': False,\n"
        "            'error': traceback.format_exc(),\n"
        "            'description': tc.get('description', '')\n"
        "        })\n\n"
        "print('__TEST_RESULT__:' + json.dumps({'results': _results}))\n"
    )
    return script
