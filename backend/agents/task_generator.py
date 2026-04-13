"""
Task Generator Agent.
Generates coding challenges for the workspace module.
Two modes: Diagnostic (probe weak areas) and Challenge (push strong candidates).
"""

import json

from backend.config import settings
from backend.utils.llm import generate_json

TASK_GENERATION_PROMPT = """Design a coding challenge.

Role: {role}
Experience: {experience_level}
Trigger: {trigger_reason}
Topic: {current_topic}
Skills: {relevant_skills}
Custom Request: {custom_request}
Recent performance:
{evaluation_summary}

Rules:
- 5-10 minutes, Python only, no external libs.
- Include 3-5 test cases and starter code.
- If 'Custom Request' is present, prioritize its requirements.

Return JSON only:
{{
    "title": "Brief descriptive title",
    "description": "Clear problem statement (2-4 paragraphs). Include:\n- What the function should do\n- Input format\n- Output format\n- Constraints\n- At least one example",
    "difficulty": "easy|medium|hard",
    "starter_code": "def solution(...):\\n    # Your code here\\n    pass",
    "test_cases": [
        {{
            "input": "the input to test",
            "expected_output": "expected result",
            "description": "what this test checks"
        }}
    ],
    "time_limit_minutes": 5-10,
    "evaluation_criteria": [
        "correctness",
        "edge case handling",
        "code quality",
        "time complexity awareness"
    ],
    "hints": ["optional hint 1"],
    "topic_tested": "specific skill being tested"
}}
"""


async def generate_task(state: dict) -> dict:
    """
    LangGraph node: Generate a coding task for the workspace.

    Args:
        state: Current interview state

    Returns:
        State update with 'workspace' and 'mode'
    """
    role = state.get("role", "Software Engineer")
    role_lc = role.lower()
    non_tech_keywords = [
        "manager", "management", "product", "program", "project",
        "operations", "ops", "marketing", "sales", "hr", "recruit",
        "finance", "account", "legal", "customer", "support"
    ]
    if any(k in role_lc for k in non_tech_keywords):
        return {
            "mode": "conversation",
            "needs_workspace": False,
            "custom_task_request": "",
        }
    profile = state.get("profile", {})
    evaluation = state.get("evaluation", {})
    interview_plan = state.get("interview_plan", [])
    current_step = state.get("current_step", 0)
    cumulative = state.get("cumulative_scores", {})

    experience_level = profile.get("experience_level", "mid")

    # Determine trigger reason
    overall_score = cumulative.get("overall", 0.5)
    if overall_score >= 0.7:
        trigger_reason = "CHALLENGE — Candidate is performing well. Generate a harder problem to differentiate and push their limits."
    else:
        trigger_reason = "DIAGNOSTIC — Candidate showed weakness in this area. Generate a targeted task to accurately assess their ability."

    # Get current topic
    current_topic = "General Programming"
    relevant_skills = []
    if current_step < len(interview_plan):
        step_data = interview_plan[current_step]
        current_topic = step_data.get("topic", "General Programming")
        relevant_skills = step_data.get("depends_on_skills", [])

    # Build evaluation summary
    eval_summary = f"""
Cumulative scores: correctness={cumulative.get('correctness', 'N/A')}, depth={cumulative.get('depth', 'N/A')}, overall={overall_score}
Last response evaluation: {json.dumps(evaluation, indent=2) if evaluation else 'N/A'}
Detected weaknesses: {evaluation.get('detected_weaknesses', [])}
Detected strengths: {evaluation.get('detected_strengths', [])}
"""

    prompt = TASK_GENERATION_PROMPT.format(
        role=role,
        experience_level=experience_level,
        trigger_reason=trigger_reason,
        current_topic=current_topic,
        relevant_skills=", ".join(relevant_skills) if relevant_skills else "general programming",
        custom_request=state.get("custom_task_request", "None"),
        evaluation_summary=eval_summary
    )

    try:
        task_data = await generate_json(
            prompt,
            temperature=0.5,
        )

        return {
            "workspace": {
                "active": True,
                "problem": task_data,
                "user_code": task_data.get("starter_code", "def solution():\n    pass"),
                "result": {},
                "evaluation": {}
            },
            "mode": "workspace",
            "needs_workspace": False,
            "custom_task_request": ""
        }

    except (json.JSONDecodeError, Exception) as e:
        # Fallback task
        fallback_task = _generate_fallback_task(role, current_topic, experience_level)
        return {
            "workspace": {
                "active": True,
                "problem": fallback_task,
                "user_code": fallback_task["starter_code"],
                "result": {},
                "evaluation": {}
            },
            "mode": "workspace",
            "needs_workspace": False,
            "custom_task_request": ""
        }


def _generate_fallback_task(role: str, topic: str, level: str) -> dict:
    """Generate a reasonable fallback coding task."""
    if level in ["junior", "unknown"]:
        return {
            "title": "Two Sum",
            "description": (
                "Given a list of integers `nums` and an integer `target`, return the indices "
                "of the two numbers that add up to `target`.\n\n"
                "You may assume that each input has exactly one solution, and you may not "
                "use the same element twice.\n\n"
                "Example:\n"
                "  Input: nums = [2, 7, 11, 15], target = 9\n"
                "  Output: [0, 1]  (because nums[0] + nums[1] = 2 + 7 = 9)"
            ),
            "difficulty": "easy",
            "starter_code": "def solution(nums: list, target: int) -> list:\n    # Your code here\n    pass",
            "test_cases": [
                {"input": "([2, 7, 11, 15], 9)", "expected_output": "[0, 1]", "description": "Basic case"},
                {"input": "([3, 2, 4], 6)", "expected_output": "[1, 2]", "description": "Non-obvious pair"},
                {"input": "([3, 3], 6)", "expected_output": "[0, 1]", "description": "Duplicate values"},
            ],
            "time_limit_minutes": 7,
            "evaluation_criteria": ["correctness", "edge cases", "efficiency"],
            "hints": ["Consider using a dictionary for O(n) lookup"],
            "topic_tested": "Problem Solving"
        }
    else:
        return {
            "title": "LRU Cache Implementation",
            "description": (
                "Design and implement a Least Recently Used (LRU) cache.\n\n"
                "Implement the class `LRUCache` with:\n"
                "- `__init__(self, capacity: int)` — Initialize with positive capacity\n"
                "- `get(self, key: int) -> int` — Return value if key exists, else -1\n"
                "- `put(self, key: int, value: int) -> None` — Update/insert. Evict LRU if at capacity.\n\n"
                "Both operations must run in O(1) time.\n\n"
                "Example:\n"
                "  cache = LRUCache(2)\n"
                "  cache.put(1, 1)\n"
                "  cache.put(2, 2)\n"
                "  cache.get(1)    # returns 1\n"
                "  cache.put(3, 3) # evicts key 2\n"
                "  cache.get(2)    # returns -1"
            ),
            "difficulty": "hard",
            "starter_code": (
                "class LRUCache:\n"
                "    def __init__(self, capacity: int):\n"
                "        # Your code here\n"
                "        pass\n\n"
                "    def get(self, key: int) -> int:\n"
                "        # Your code here\n"
                "        pass\n\n"
                "    def put(self, key: int, value: int) -> None:\n"
                "        # Your code here\n"
                "        pass\n"
            ),
            "test_cases": [
                {"input": "LRUCache(2); put(1,1); put(2,2); get(1)", "expected_output": "1", "description": "Basic get"},
                {"input": "LRUCache(2); put(1,1); put(2,2); put(3,3); get(2)", "expected_output": "-1", "description": "Eviction"},
                {"input": "LRUCache(1); put(1,1); put(2,2); get(1)", "expected_output": "-1", "description": "Capacity 1"},
            ],
            "time_limit_minutes": 10,
            "evaluation_criteria": ["correctness", "O(1) operations", "code structure", "edge cases"],
            "hints": ["Consider OrderedDict or doubly-linked list + hashmap"],
            "topic_tested": "Data Structures"
        }
