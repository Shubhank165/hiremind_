"""
Evaluation Agent.
Scores each candidate response on multiple dimensions.
Uses chain-of-thought reasoning before scoring to avoid fake/lazy scoring.
Updates cumulative scores and detects strengths/weaknesses dynamically.
"""

import json

from backend.utils.llm import generate_json

EVALUATION_PROMPT = """Role: {role}
Topic: {current_topic}
Question: {question}
Answer: {response}

Candidate profile:
- experience_level: {experience_level}
- relevant_skills: {relevant_skills}

Previous scores:
{cumulative_scores}

Rules:
- Be honest. Mediocre = 0.4-0.5.
- If weak/vague, probe same topic.

Return JSON only with this schema:
{{
    "reasoning": "your step-by-step analysis (2-3 sentences)",
    "correctness": 0.0-1.0,
    "depth": 0.0-1.0,
    "clarity": 0.0-1.0,
    "confidence": 0.0-1.0,
    "feedback": "internal note for the conversation agent on what to do next",
    "detected_strengths": ["strength identified in this response"],
    "detected_weaknesses": ["weakness identified in this response"],
    "should_probe_deeper": true/false,
    "should_trigger_workspace": true/false
}}
"""


async def evaluate_response(state: dict) -> dict:
    """
    LangGraph node: Evaluate the candidate's latest response.

    Args:
        state: Current interview state

    Returns:
        State update with 'evaluation', 'cumulative_scores', 'needs_workspace', 'probe_deeper'
    """
    role = state.get("role", "Software Engineer")
    profile = state.get("profile", {})
    interview_plan = state.get("interview_plan", [])
    current_step = state.get("current_step", 0)
    question = state.get("current_question", "")
    response_text = state.get("last_user_response", "")
    cumulative = state.get("cumulative_scores", {})

    if not response_text.strip():
        return {
            "evaluation": {
                "correctness": 0.0,
                "depth": 0.0,
                "clarity": 0.0,
                "confidence": 0.0,
                "feedback": "No response provided",
                "detected_strengths": [],
                "detected_weaknesses": ["No response given"],
                "should_probe_deeper": False,
                "should_trigger_workspace": False
            },
            "probe_deeper": False,
            "needs_workspace": False
        }

    # Get current topic
    current_topic = "General"
    relevant_skills = []
    if current_step < len(interview_plan):
        step_data = interview_plan[current_step]
        current_topic = step_data.get("topic", "General")
        relevant_skills = step_data.get("depends_on_skills", [])

    prompt = EVALUATION_PROMPT.format(
        role=role,
        current_topic=current_topic,
        question=question,
        response=response_text,
        experience_level=profile.get("experience_level", "unknown"),
        relevant_skills=", ".join(relevant_skills) if relevant_skills else "general",
        cumulative_scores=json.dumps(cumulative, indent=2)
    )

    try:
        evaluation = await generate_json(
            prompt,
            temperature=0.2,
        )

        # Update cumulative scores
        n = cumulative.get("num_evaluations", 0)
        new_cumulative = {
            "correctness": _running_avg(cumulative.get("correctness", 0), evaluation.get("correctness", 0.5), n),
            "depth": _running_avg(cumulative.get("depth", 0), evaluation.get("depth", 0.5), n),
            "clarity": _running_avg(cumulative.get("clarity", 0), evaluation.get("clarity", 0.5), n),
            "confidence": _running_avg(cumulative.get("confidence", 0), evaluation.get("confidence", 0.5), n),
            "num_evaluations": n + 1
        }
        new_cumulative["overall"] = (
            new_cumulative["correctness"] * 0.35 +
            new_cumulative["depth"] * 0.30 +
            new_cumulative["clarity"] * 0.20 +
            new_cumulative["confidence"] * 0.15
        )

        return {
            "evaluation": evaluation,
            "cumulative_scores": new_cumulative,
            "probe_deeper": evaluation.get("should_probe_deeper", False),
            "needs_workspace": evaluation.get("should_trigger_workspace", False)
        }

    except (json.JSONDecodeError, Exception) as e:
        # Fallback neutral evaluation
        return {
            "evaluation": {
                "correctness": 0.5,
                "depth": 0.5,
                "clarity": 0.5,
                "confidence": 0.5,
                "feedback": f"Evaluation parsing error: {str(e)}",
                "detected_strengths": [],
                "detected_weaknesses": [],
                "should_probe_deeper": False,
                "should_trigger_workspace": False
            },
            "probe_deeper": False,
            "needs_workspace": False
        }


def _running_avg(current_avg: float, new_value: float, n: int) -> float:
    """Calculate running average incrementally."""
    if n == 0:
        return new_value
    return round((current_avg * n + new_value) / (n + 1), 4)
