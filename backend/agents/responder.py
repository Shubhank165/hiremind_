"""
Combined Respond-and-Evaluate Agent.
Merges evaluation + question generation into a SINGLE LLM call.
This is the key optimization: 1 call instead of 2-3 per turn.
"""

import json
from datetime import datetime, timezone
from google import genai
from backend.config import settings

client = genai.Client(api_key=settings.gemini_api_key)

COMBINED_PROMPT = """You are a senior technical interviewer for the role of {role}. You must BOTH evaluate the candidate's last answer AND generate your next response in a single step.

## Rules:
- Ask ONE clear question at a time
- Keep your spoken response SHORT (2-4 sentences max)
- Score HONESTLY — a mediocre answer is 0.4-0.5, not 0.7
- If the answer is weak/vague, probe deeper on the SAME topic
- If the answer is strong, move to the next planned topic
- Never reveal scores to the candidate
- Be natural — like a real interviewer, not a bot

## Candidate Profile:
Experience: {experience_level}
Summary: {profile_summary}

## Current Plan Step ({step_num}/{total_steps}):
{current_plan_step}

## Question You Asked:
{question}

## Candidate's Answer:
{answer}

## Recent Context (last 3 exchanges):
{recent_context}

## Output STRICT JSON:
{{
    "evaluation": {{
        "correctness": 0.0,
        "depth": 0.0,
        "clarity": 0.0,
        "confidence": 0.0
    }},
    "action": "continue|probe|workspace|end",
    "next_response": "Your natural spoken interviewer response including the next question",
    "detected_strengths": [],
    "detected_weaknesses": []
}}

Action guide:
- "continue" = answer was adequate, move to next topic
- "probe" = answer was weak/vague, ask follow-up on SAME topic
- "workspace" = candidate needs a coding challenge to verify this skill
- "end" = all planned topics covered"""

FIRST_QUESTION_PROMPT = """You are a senior technical interviewer starting an interview for {role}.

Candidate Profile:
{profile_summary}

First planned topic: {first_topic}

Generate a warm but professional opening (3-4 sentences). Welcome the candidate, briefly introduce yourself, and ask your first question.

Output ONLY the spoken words — no JSON, no labels."""


async def generate_first_question(state: dict) -> dict:
    """Generate the opening question. Single LLM call."""
    role = state.get("role", "Software Engineer")
    profile = state.get("profile", {})
    plan = state.get("interview_plan", [])

    first_topic = json.dumps(plan[0], indent=2) if plan else '{"topic": "Introduction"}'

    try:
        response = await client.aio.models.generate_content(
            model=settings.gemini_model,
            contents=FIRST_QUESTION_PROMPT.format(
                role=role,
                profile_summary=profile.get("summary", "No profile"),
                first_topic=first_topic
            ),
            config=genai.types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=250,
            )
        )

        question = response.text.strip()
        entry = {
            "role": "interviewer",
            "content": question,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": 0
        }
        return {
            "current_question": question,
            "conversation_history": [entry],
            "mode": "conversation"
        }
    except Exception as e:
        fallback = f"Hello! Welcome to this interview for the {role} position. Could you start by telling me about your background and what interests you about this role?"
        return {
            "current_question": fallback,
            "conversation_history": [{"role": "interviewer", "content": fallback, "timestamp": datetime.now(timezone.utc).isoformat(), "step": 0}],
            "mode": "conversation"
        }


async def respond_and_evaluate(state: dict) -> dict:
    """
    COMBINED node: evaluate response + generate next question in ONE LLM call.
    This replaces 3 separate nodes (process_input, evaluate, generate_question).
    """
    role = state.get("role", "Software Engineer")
    profile = state.get("profile", {})
    plan = state.get("interview_plan", [])
    current_step = state.get("current_step", 0)
    question = state.get("current_question", "")
    answer = state.get("last_user_response", "")
    history = state.get("conversation_history", [])
    cumulative = state.get("cumulative_scores", {})

    # Add user response to history
    user_entry = {
        "role": "candidate",
        "content": answer,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "step": current_step
    }

    # Build context
    recent = history[-6:] if len(history) > 6 else history
    recent_text = "\n".join([
        f"{'Interviewer' if m.get('role') == 'interviewer' else 'Candidate'}: {m.get('content', '')[:150]}"
        for m in recent if isinstance(m, dict)
    ]) or "Start of interview"

    current_plan_step = json.dumps(plan[current_step], indent=2) if current_step < len(plan) else '{"topic": "Wrap-up"}'

    prompt = COMBINED_PROMPT.format(
        role=role,
        experience_level=profile.get("experience_level", "unknown"),
        profile_summary=profile.get("summary", "No profile")[:200],
        step_num=current_step + 1,
        total_steps=len(plan),
        current_plan_step=current_plan_step,
        question=question,
        answer=answer,
        recent_context=recent_text
    )

    try:
        response = await client.aio.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.4,
                response_mime_type="application/json",
                max_output_tokens=1500,
            )
        )

        result_text = response.text.strip()
        if result_text.startswith("```"):
            result_text = result_text.split("\n", 1)[1]
            if result_text.endswith("```"):
                result_text = result_text[:-3]

        data = json.loads(result_text.strip())

        # Extract evaluation
        eval_data = data.get("evaluation", {})
        action = data.get("action", "continue")
        next_response = data.get("next_response", "Could you elaborate on that?")

        # Update cumulative scores
        n = cumulative.get("num_evaluations", 0)
        new_cumulative = {
            "correctness": _ravg(cumulative.get("correctness", 0), eval_data.get("correctness", 0.5), n),
            "depth": _ravg(cumulative.get("depth", 0), eval_data.get("depth", 0.5), n),
            "clarity": _ravg(cumulative.get("clarity", 0), eval_data.get("clarity", 0.5), n),
            "confidence": _ravg(cumulative.get("confidence", 0), eval_data.get("confidence", 0.5), n),
            "num_evaluations": n + 1
        }
        new_cumulative["overall"] = (
            new_cumulative["correctness"] * 0.35 +
            new_cumulative["depth"] * 0.30 +
            new_cumulative["clarity"] * 0.20 +
            new_cumulative["confidence"] * 0.15
        )

        # Build interviewer history entry
        interviewer_entry = {
            "role": "interviewer",
            "content": next_response,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": current_step
        }

        # Decide next step
        should_advance = action == "continue"
        needs_workspace = action == "workspace"
        should_end = action == "end" or current_step >= len(plan) - 1

        new_step = current_step + 1 if should_advance else current_step

        return {
            "conversation_history": [user_entry, interviewer_entry],
            "current_question": next_response,
            "evaluation": {
                **eval_data,
                "feedback": f"Action: {action}",
                "detected_strengths": data.get("detected_strengths", []),
                "detected_weaknesses": data.get("detected_weaknesses", []),
                "should_probe_deeper": action == "probe",
                "should_trigger_workspace": needs_workspace,
            },
            "cumulative_scores": new_cumulative,
            "current_step": new_step,
            "needs_workspace": needs_workspace,
            "should_end": should_end,
            "probe_deeper": action == "probe",
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"RESPOND_ERROR: {e}")
        fallback = "That's an interesting point. Could you tell me more about your approach?"
        interviewer_entry = {
            "role": "interviewer",
            "content": fallback,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": current_step
        }
        return {
            "conversation_history": [user_entry, interviewer_entry],
            "current_question": fallback,
            "evaluation": {"correctness": 0.5, "depth": 0.5, "clarity": 0.5, "confidence": 0.5},
            "cumulative_scores": cumulative,
            "current_step": current_step,
            "needs_workspace": False,
            "should_end": False,
            "probe_deeper": False,
        }


def _ravg(current: float, new_val: float, n: int) -> float:
    if n == 0:
        return round(new_val, 4)
    return round((current * n + new_val) / (n + 1), 4)
