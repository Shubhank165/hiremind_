"""
LangGraph Orchestrator — OPTIMIZED.
Uses combined respond_and_evaluate node for 1 LLM call per turn instead of 2-3.
"""

import json
from typing import Literal
from datetime import datetime, timezone
from langgraph.graph import StateGraph, START, END

from backend.state import InterviewState
from backend.agents.profile_extractor import extract_profile
from backend.agents.interview_planner import generate_interview_plan
from backend.agents.responder import respond_and_evaluate, generate_first_question
from backend.agents.task_generator import generate_task
from backend.config import settings
from google import genai

client = genai.Client(api_key=settings.gemini_api_key)


# ────────────────────────────────────────────
# Node: Evaluate workspace submission
# ────────────────────────────────────────────
async def evaluate_workspace(state: dict) -> dict:
    """Evaluate workspace code submission."""
    workspace = state.get("workspace", {})
    problem = workspace.get("problem", {})
    user_code = workspace.get("user_code", "")
    result = workspace.get("result", {})
    role = state.get("role", "Software Engineer")

    eval_prompt = f"""Evaluate this coding solution briefly for a {role} interview.

Problem: {problem.get('title', 'Unknown')}
Code:
```python
{user_code[:1000]}
```
Tests: {result.get('passed_tests', 0)}/{result.get('total_tests', 0)} passed
Errors: {result.get('stderr', '')[:300]}

Output JSON: {{"correctness": 0-1, "code_quality": 0-1, "overall": 0-1, "feedback": "1 sentence"}}"""

    try:
        response = await client.aio.models.generate_content(
            model=settings.gemini_model,
            contents=eval_prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
                max_output_tokens=200,
            )
        )
        ws_eval = json.loads(response.text.strip())
    except Exception:
        ws_eval = {"correctness": 0.5, "code_quality": 0.5, "overall": 0.4, "feedback": "Auto-scored"}

    updated_workspace = {**workspace, "evaluation": ws_eval, "active": False}
    return {"workspace": updated_workspace, "mode": "conversation"}


# ────────────────────────────────────────────
# Node: Generate final report
# ────────────────────────────────────────────
async def generate_final_report(state: dict) -> dict:
    """Generate final interview assessment."""
    role = state.get("role", "Software Engineer")
    profile = state.get("profile", {})
    cumulative = state.get("cumulative_scores", {})
    history = state.get("conversation_history", [])
    workspace = state.get("workspace", {})

    conv_summary = "\n".join([
        f"{'Q' if m.get('role')=='interviewer' else 'A'}: {m.get('content','')[:150]}"
        for m in history[-16:] if isinstance(m, dict)
    ])

    report_prompt = f"""Generate interview assessment for {role}.
Profile: {json.dumps(profile, indent=1)[:800]}
Scores: {json.dumps(cumulative)}
Workspace: {json.dumps(workspace.get('evaluation', {}))}
Transcript summary:
{conv_summary}

Output JSON:
{{"overall_score": 0-1, "recommendation": "strong_hire|hire|borderline|no_hire",
"strengths": ["..."], "weaknesses": ["..."],
"skill_breakdown": {{"skill": {{"score": 0-1, "notes": "brief"}}}},
"detailed_feedback": "2-3 paragraphs",
"suggested_next_steps": "1 sentence"}}"""

    try:
        response = await client.aio.models.generate_content(
            model=settings.gemini_model,
            contents=report_prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.3,
                response_mime_type="application/json",
            )
        )
        report = json.loads(response.text.strip())
    except Exception:
        overall = cumulative.get("overall", 0.5)
        rec = "hire" if overall >= 0.7 else "borderline" if overall >= 0.5 else "no_hire"
        report = {
            "overall_score": overall,
            "recommendation": rec,
            "strengths": profile.get("strengths", []),
            "weaknesses": profile.get("weaknesses", []),
            "skill_breakdown": {},
            "detailed_feedback": "Automated report based on cumulative scores.",
            "suggested_next_steps": ""
        }

    return {"final_report": report, "mode": "complete", "should_end": True}


# ────────────────────────────────────────────
# Routing
# ────────────────────────────────────────────
def route_after_combined(state: dict) -> Literal["__end__", "generate_task", "generate_final_report"]:
    """Route after combined respond+evaluate. Most turns go straight to END."""
    if state.get("should_end", False):
        return "generate_final_report"
    if state.get("needs_workspace", False):
        return "generate_task"
    # Normal case: question already generated inline — done!
    return "__end__"


# ────────────────────────────────────────────
# Graph: Init (profile → plan → first question)
# ────────────────────────────────────────────
def build_init_graph():
    builder = StateGraph(InterviewState)
    builder.add_node("extract_profile", extract_profile)
    builder.add_node("generate_plan", generate_interview_plan)
    builder.add_node("first_question", generate_first_question)

    builder.add_edge(START, "extract_profile")
    builder.add_edge("extract_profile", "generate_plan")
    builder.add_edge("generate_plan", "first_question")
    builder.add_edge("first_question", END)
    return builder.compile()


# ────────────────────────────────────────────
# Graph: Conversation (1 LLM call per turn!)
# ────────────────────────────────────────────
def build_conversation_graph():
    """
    Optimized: START → respond_and_evaluate → route →
        ├── END (most turns — question already generated)
        ├── generate_task → END
        └── generate_final_report → END
    """
    builder = StateGraph(InterviewState)

    builder.add_node("respond_and_evaluate", respond_and_evaluate)
    builder.add_node("generate_task", generate_task)
    builder.add_node("generate_final_report", generate_final_report)

    builder.add_edge(START, "respond_and_evaluate")

    builder.add_conditional_edges(
        "respond_and_evaluate",
        route_after_combined,
        {
            "__end__": END,
            "generate_task": "generate_task",
            "generate_final_report": "generate_final_report",
        }
    )

    builder.add_edge("generate_task", END)
    builder.add_edge("generate_final_report", END)

    return builder.compile()


# ────────────────────────────────────────────
# Graph: Workspace eval
# ────────────────────────────────────────────
def build_workspace_graph():
    builder = StateGraph(InterviewState)
    builder.add_node("evaluate_workspace", evaluate_workspace)
    builder.add_node("first_question", generate_first_question)

    builder.add_edge(START, "evaluate_workspace")
    builder.add_edge("evaluate_workspace", "first_question")
    builder.add_edge("first_question", END)
    return builder.compile()


# Pre-compiled singleton graphs
init_graph = build_init_graph()
conversation_graph = build_conversation_graph()
workspace_graph = build_workspace_graph()
