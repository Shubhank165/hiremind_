"""
Combined Respond-and-Evaluate Agent.
Merges evaluation + question generation into a SINGLE LLM call.
This is the key optimization: 1 call instead of 2-3 per turn.
"""

import json
from datetime import datetime, timezone

from backend.utils.llm import generate_json, generate_text

COMBINED_PROMPT = """ROLE: Senior technical interviewer for {role}.
TASK: Evaluate last answer AND generate next response in one step.

Tone:
- Warm, human, and professional. Use brief greetings or acknowledgments.
- Avoid blunt or scolding phrasing like "you're off topic". If needed, gently steer back.
- If the candidate says hello or greets you, reply with a brief hello and proceed.

Rules:
- Ask exactly ONE question.
- Keep response 2-4 sentences.
- If answer weak, give a lower evaluation and ask at most ONE follow-up, then move on.
- If strong, advance to next plan step.
- Never reveal scores.
- If profile has projects, reference them before generic topics.

Candidate profile:
- experience_level: {experience_level}
- summary: {profile_summary}

Plan step ({step_num}/{total_steps}):
{current_plan_step}

Question asked:
{question}

Answer:
{answer}

Recent context:
{recent_context}

Return JSON only with this schema:
{{
    "evaluation": {{
        "correctness": 0.0,
        "depth": 0.0,
        "clarity": 0.0,
        "confidence": 0.0
    }},
    "action": "continue|probe|workspace|end",
    "next_response": "Your natural spoken interviewer response including the next question",
    "custom_task_request": "Optional: Specific details or constraints for a coding task if action is 'workspace'",
    "detected_strengths": [],
    "detected_weaknesses": []
}}

Action guide:
- continue = move to next topic
- probe = follow-up on same topic
- workspace = coding challenge. Use 'custom_task_request' to specify what the candidate should build.
- end = all planned topics covered"""

FIRST_QUESTION_PROMPT = """You are a senior technical interviewer for {role}.

Start with a warm greeting and a quick intro (you can use a generic name), then ask a high-level background question.
Candidate profile: {profile_summary}
First plan topic: {first_topic}

Output 2-4 sentences. Do NOT dive into detailed implementation topics on the first question.
Avoid generic data-structure questions unless profile is empty."""


async def generate_first_question(state: dict) -> dict:
    """Generate the opening question. Single LLM call."""
    role = state.get("role", "Software Engineer")
    profile = state.get("profile", {})
    plan = state.get("interview_plan", [])

    first_topic = json.dumps(plan[0], indent=2) if plan else '{"topic": "Introduction"}'

    try:
        question = await generate_text(
            FIRST_QUESTION_PROMPT.format(
                role=role,
                profile_summary=profile.get("summary", "No profile"),
                first_topic=first_topic,
            ),
            temperature=0.7,
        )
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
        data = await generate_json(
            prompt,
            temperature=0.4,
        )

        # Extract evaluation
        eval_data = data.get("evaluation", {})
        action = data.get("action", "continue")
        next_response = data.get("next_response", "Could you elaborate on that?")

        step_type = None
        if current_step < len(plan) and isinstance(plan[current_step], dict):
            step_type = plan[current_step].get("type")

        role_lc = role.lower()
        non_tech_keywords = [
            "manager", "management", "product", "program", "project",
            "operations", "ops", "marketing", "sales", "hr", "recruit",
            "finance", "account", "legal", "customer", "support"
        ]
        is_technical = not any(k in role_lc for k in non_tech_keywords)

        if is_technical and step_type == "coding" and not state.get("workspace", {}).get("active", False):
            action = "workspace"
            next_response = (
                "Great. For the final step, we will do a short coding assessment. "
                "We will go through requirements, edge cases, optimize, and refactor. "
                "Please complete the task and submit your solution when you're ready."
            )

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
        questions_in_step = sum(
            1 for m in history
            if isinstance(m, dict) and m.get("role") == "interviewer" and m.get("step") == current_step
        )
        # Keep each plan step to 1-2 questions max.
        force_advance = questions_in_step >= 1
        should_advance = action == "continue" or (action == "probe" and force_advance)
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
            "custom_task_request": data.get("custom_task_request", ""),
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


async def acknowledge_workspace(state: dict) -> dict:
    """Generate response after workspace task completion."""
    role = state.get("role", "Software Engineer")
    workspace = state.get("workspace", {})
    problem = workspace.get("problem", {})
    eval_data = workspace.get("evaluation", {})
    plan = state.get("interview_plan", [])
    current_step = state.get("current_step", 0)

    # Advance step
    new_step = current_step + 1
    next_topic = plan[new_step].get("topic", "Next Topic") if new_step < len(plan) else "Final Thoughts"

    prompt = f"""You are a senior technical interviewer for {role}.
The candidate just finished a coding task: {problem.get('title')}.
Evaluation: {eval_data.get('feedback', 'Completed')}
Passed tests: {workspace.get('result', {}).get('passed_tests', 0)}/{workspace.get('result', {}).get('total_tests', 0)}

Task: Briefly (1-2 sentences) acknowledge their work/approach and then transition to the next topic: {next_topic}.
Ask a relevant opening question for the new topic."""

    try:
        response = await generate_text(prompt, temperature=0.7)
        entry = {
            "role": "interviewer",
            "content": response,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": new_step
        }
        return {
            "current_question": response,
            "conversation_history": [entry],
            "current_step": new_step,
            "mode": "conversation"
        }
    except Exception:
        fallback = f"Thanks for working through that coding challenge. Let's move on to discuss {next_topic}. Can you tell me more about your experience with it?"
        return {
            "current_question": fallback,
            "conversation_history": [{"role": "interviewer", "content": fallback, "timestamp": datetime.now(timezone.utc).isoformat(), "step": new_step}],
            "current_step": new_step,
            "mode": "conversation"
        }


def _ravg(current: float, new_val: float, n: int) -> float:
    if n == 0:
        return round(new_val, 4)
    return round((current * n + new_val) / (n + 1), 4)
