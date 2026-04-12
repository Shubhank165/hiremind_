"""
Global interview state definition for LangGraph orchestration.
This TypedDict is the single source of truth for the entire interview session.
"""

import operator
from typing import TypedDict, Annotated


class WorkspaceState(TypedDict):
    """State for the coding workspace module."""
    active: bool
    problem: dict       # {title, description, difficulty, starter_code, test_cases, time_limit, criteria}
    user_code: str
    result: dict        # {stdout, stderr, passed_tests, total_tests, execution_time}
    evaluation: dict    # workspace-specific evaluation


class InterviewState(TypedDict):
    """
    The complete interview state that flows through the LangGraph orchestrator.
    Every agent reads from and writes to subsets of this state.
    """
    # Session setup
    role: str
    resume_data: str      # Base64 encoded file data
    resume_mime_type: str # e.g. application/pdf

    # Profile Intelligence
    profile: dict
    # {
    #   "skills": [{"name": str, "level": str, "confidence": float}],
    #   "experience_level": str,
    #   "projects": [{"name": str, "description": str, "technologies": list, "impact": str}],
    #   "strengths": [str],
    #   "weaknesses": [str],
    #   "confidence_scores": {str: float}
    # }

    # Interview Planning
    interview_plan: list
    # [{"step": int, "topic": str, "type": str, "difficulty": str,
    #   "duration_minutes": int, "objective": str, "depends_on_skills": list}]

    # Conversation tracking
    current_step: int
    current_question: str
    conversation_history: Annotated[list[dict], operator.add]
    # Each entry: {"role": "interviewer"|"candidate", "content": str, "timestamp": str}
    last_user_response: str

    # Evaluation
    evaluation: dict
    # Per-response: {"correctness": float, "depth": float, "clarity": float,
    #                "confidence": float, "feedback": str,
    #                "detected_strengths": list, "detected_weaknesses": list,
    #                "should_probe_deeper": bool, "should_trigger_workspace": bool}

    cumulative_scores: dict
    # Running averages: {"correctness": float, "depth": float, "clarity": float,
    #                     "confidence": float, "overall": float}

    # Mode control
    mode: str           # "init" | "profiling" | "planning" | "conversation" | "workspace" | "complete"

    # Workspace
    workspace: dict     # WorkspaceState equivalent as dict for LangGraph compatibility

    # Final output
    final_report: dict
    # {"overall_score": float, "strengths": list, "weaknesses": list,
    #  "skill_breakdown": dict, "recommendation": str, "detailed_feedback": str}

    # Control flags
    should_end: bool
    needs_workspace: bool
    probe_deeper: bool


def create_initial_state(role: str, resume_data: str = "", resume_mime_type: str = "text/plain") -> dict:
    """Create a fresh interview state with defaults."""
    return {
        "role": role,
        "resume_data": resume_data,
        "resume_mime_type": resume_mime_type,
        "profile": {},
        "interview_plan": [],
        "current_step": 0,
        "current_question": "",
        "conversation_history": [],
        "last_user_response": "",
        "evaluation": {},
        "cumulative_scores": {
            "correctness": 0.0,
            "depth": 0.0,
            "clarity": 0.0,
            "confidence": 0.0,
            "overall": 0.0,
            "num_evaluations": 0
        },
        "mode": "init",
        "workspace": {
            "active": False,
            "problem": {},
            "user_code": "",
            "result": {},
            "evaluation": {}
        },
        "final_report": {},
        "should_end": False,
        "needs_workspace": False,
        "probe_deeper": False,
    }
