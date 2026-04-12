"""
Interview Planner Agent.
Uses the extracted profile to generate a personalized, adaptive interview plan.
Plans are NOT static — they can be regenerated mid-interview based on evaluation data.
"""

import json
from google import genai
from backend.config import settings

client = genai.Client(api_key=settings.gemini_api_key)

PLANNING_PROMPT = """You are an expert technical interview architect. Design a personalized interview plan based on the candidate's profile and target role.

## Target Role: {role}

## Candidate Profile:
{profile}

## Evaluation Data So Far (if any):
{evaluation_data}

## Design Principles:
1. **Adaptive difficulty**: Start at the candidate's apparent level, then ramp up
2. **Weakness probing**: Dedicate steps to areas where confidence is low or gaps exist
3. **Strength validation**: Include steps that let strong candidates demonstrate mastery
4. **Workspace checkpoints**: Include 1-2 coding/practical tasks at strategic points
5. **Natural flow**: Topics should transition smoothly, not feel like random questions
6. **Time-aware**: Total interview should be {min_steps}-{max_steps} steps

## Step Types:
- "conceptual" — test understanding of concepts/theory
- "practical" — test ability to solve problems or design systems
- "behavioral" — test past experience, decision-making, collaboration
- "coding" — hands-on workspace task (triggers workspace mode)
- "system_design" — whiteboard-style system design discussion

## Output Format (JSON only, no markdown):
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
        response = await client.aio.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.4,
                response_mime_type="application/json",
            )
        )

        plan_text = response.text.strip()
        if plan_text.startswith("```"):
            plan_text = plan_text.split("\n", 1)[1]
            if plan_text.endswith("```"):
                plan_text = plan_text[:-3]
            plan_text = plan_text.strip()

        plan_data = json.loads(plan_text)
        interview_plan = plan_data.get("interview_plan", [])

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
