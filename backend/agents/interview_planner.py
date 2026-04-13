"""
Interview Planner Agent.
Uses the extracted profile to generate a personalized, adaptive interview plan.
Plans are NOT static — they can be regenerated mid-interview based on evaluation data.
"""

import json

from backend.config import settings
from backend.utils.llm import generate_json

PLANNING_PROMPT = """Design a concise interview plan suitable for a 5-minute demo.

Role: {role}
Profile: {profile}
Eval data: {evaluation_data}

-Rules:
- Use EXACTLY 5 steps. Do not return more or fewer.
- Start with 1-2 warm-up steps (background, role motivation, high-level experience).
- Avoid deep implementation details until after the warm-up.
- Prefer project-specific topics if available, but only after the warm-up.
- Only include coding tasks for technical roles (software, data, ML, engineering). For management or business roles, do NOT include coding tasks.
- If you include a coding task, make it the FINAL step.
- For management/business roles, the final step should be a wrap-up or leadership topic, not coding.

Return JSON only:
{{
    "interview_plan": [
        {{
            "step": 1,
            "topic": "specific topic name",
            "type": "conceptual|practical|behavioral|coding|system_design",
            "difficulty": "easy|medium|hard",
            "duration_minutes": 3-10,
            "objective": "what we want to learn from this step",
            "depends_on_skills": ["skill1", "skill2"],
            "follow_up_strategy": "probe deeper if weak | advance if strong | always ask"
        }}
    ],
    "plan_rationale": "brief explanation of why this plan was designed this way",
    "difficulty_curve": "description of how difficulty ramps",
    "total_estimated_minutes": 30-60
}}
"""


async def generate_interview_plan(state: dict) -> dict:
    """
    LangGraph node: Generate a personalized interview plan.

    Args:
        state: Current interview state with 'profile' and 'role'

    Returns:
        State update with 'interview_plan' and 'mode'
    """
    profile = state.get("profile", {})
    role = state.get("role", "Software Engineer")
    evaluation_data = state.get("cumulative_scores", {})

    prompt = PLANNING_PROMPT.format(
        role=role,
        profile=json.dumps(profile, indent=2),
        evaluation_data=json.dumps(evaluation_data, indent=2) if evaluation_data.get("num_evaluations", 0) > 0 else "No evaluations yet — this is the initial plan.",
        min_steps=settings.min_interview_steps,
        max_steps=settings.max_interview_steps
    )

    try:
        plan_data = await generate_json(
            prompt,
            temperature=0.4,
        )
        interview_plan = plan_data.get("interview_plan", [])

        if len(interview_plan) > settings.max_interview_steps:
            interview_plan = interview_plan[:settings.max_interview_steps]

        # Ensure steps are properly numbered
        for i, step in enumerate(interview_plan):
            step["step"] = i + 1

        return {
            "interview_plan": interview_plan,
            "mode": "conversation",
            "current_step": 0
        }

    except (json.JSONDecodeError, Exception) as e:
        # Fallback: generate a sensible default plan
        default_plan = _generate_fallback_plan(role, profile)
        return {
            "interview_plan": default_plan,
            "mode": "conversation",
            "current_step": 0
        }


def _generate_fallback_plan(role: str, profile: dict) -> list:
    """Generate a reasonable default plan if LLM planning fails."""
    experience = profile.get("experience_level", "mid")
    skills = [s.get("name", "") for s in profile.get("skills", [])][:5]
    weaknesses = profile.get("weaknesses", [])[:3]

    base_difficulty = "easy" if experience in ["junior", "unknown"] else "medium"

    plan = [
        {
            "step": 1,
            "topic": "Introduction and Background",
            "type": "behavioral",
            "difficulty": "easy",
            "duration_minutes": 3,
            "objective": "Understand candidate's background and motivation",
            "depends_on_skills": [],
            "follow_up_strategy": "always ask"
        },
        {
            "step": 2,
            "topic": f"Core {role} Concepts",
            "type": "conceptual",
            "difficulty": base_difficulty,
            "duration_minutes": 5,
            "objective": "Assess foundational knowledge for the role",
            "depends_on_skills": skills[:2],
            "follow_up_strategy": "probe deeper if weak"
        },
    ]

    # Add skill-specific steps
    for i, skill in enumerate(skills[:3]):
        plan.append({
            "step": len(plan) + 1,
            "topic": skill,
            "type": "practical",
            "difficulty": "medium",
            "duration_minutes": 5,
            "objective": f"Assess depth in {skill}",
            "depends_on_skills": [skill],
            "follow_up_strategy": "probe deeper if weak"
        })

    # Add weakness probing
    for weakness in weaknesses[:2]:
        plan.append({
            "step": len(plan) + 1,
            "topic": weakness,
            "type": "conceptual",
            "difficulty": base_difficulty,
            "duration_minutes": 5,
            "objective": f"Assess gap: {weakness}",
            "depends_on_skills": [],
            "follow_up_strategy": "probe deeper if weak"
        })

    # Add coding task
    plan.append({
        "step": len(plan) + 1,
        "topic": f"Coding Challenge — {role}",
        "type": "coding",
        "difficulty": "medium",
        "duration_minutes": 10,
        "objective": "Assess hands-on coding ability",
        "depends_on_skills": skills[:3],
        "follow_up_strategy": "always ask"
    })

    # Add closing
    plan.append({
        "step": len(plan) + 1,
        "topic": "System Design & Wrap-up",
        "type": "system_design",
        "difficulty": "medium" if experience != "junior" else "easy",
        "duration_minutes": 5,
        "objective": "Assess system-level thinking and close interview",
        "depends_on_skills": skills,
        "follow_up_strategy": "advance if strong"
    })

    return plan
