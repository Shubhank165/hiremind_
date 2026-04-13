"""
Conversation Agent — the Interviewer Persona.
Generates contextual, adaptive questions. NOT template-based.
Behaves like a real senior technical interviewer: professional, natural, controlled.
"""

import json
from datetime import datetime, timezone

from backend.utils.llm import generate_text

INTERVIEWER_SYSTEM_PROMPT = """You are a senior technical interviewer conducting a real interview for the role of {role}. You are professional, articulate, and adaptive.

## Your Personality:
- Warm but focused — you're not a chatbot, you're evaluating
- You acknowledge good answers briefly before moving on
- You probe weak answers with targeted follow-ups
- You transition smoothly between topics
- You never repeat questions verbatim
- You speak naturally, as in a real conversation

## STRICT Rules:
1. Ask ONE clear question at a time
2. Never reveal evaluation scores to the candidate
3. Never say "great answer" unless genuinely warranted — be honest
4. If the candidate gives a vague answer, ask them to be specific
5. If the candidate clearly doesn't know something, briefly acknowledge and move on — don't humiliate
6. Keep responses SHORT (2-4 sentences max: brief context/transition + question)
"""

QUESTION_GENERATION_PROMPT = """## Current Interview Context

**Role**: {role}
**Candidate Profile Summary**: {profile_summary}

**Current Plan Step**:
{current_plan_step}

**Recent Conversation** (last 4 exchanges):
{recent_history}

**Last Evaluation** (internal — not shared with candidate):
{last_evaluation}

**Instruction**: {instruction}

Generate your next interviewer response. This should be natural speech — something a real interviewer would say out loud. Include a brief transition from the previous exchange and then your question.

Output ONLY the interviewer's spoken words. No JSON, no labels, no formatting.
"""

FIRST_QUESTION_PROMPT = """## Interview Starting

**Role**: {role}
**Candidate Profile Summary**: {profile_summary}
**First Plan Step**: {first_step}

Generate a warm but professional opening. Introduce yourself briefly (you can use a generic name), welcome the candidate, and ask your first question based on the plan step.

Keep it to 3-4 sentences. Output ONLY the interviewer's spoken words.
"""


async def generate_question(state: dict) -> dict:
    """
    LangGraph node: Generate the next interviewer question/response.

    Args:
        state: Current interview state

    Returns:
        State update with 'current_question', 'conversation_history', 'mode'
    """
    role = state.get("role", "Software Engineer")
    profile = state.get("profile", {})
    interview_plan = state.get("interview_plan", [])
    current_step = state.get("current_step", 0)
    conversation_history = state.get("conversation_history", [])
    evaluation = state.get("evaluation", {})
    probe_deeper = state.get("probe_deeper", False)

    profile_summary = profile.get("summary", "No profile available")

    # Determine if this is the first question
    is_first = len(conversation_history) == 0

    if is_first:
        first_step = interview_plan[0] if interview_plan else {"topic": "Introduction", "type": "behavioral"}
        prompt = FIRST_QUESTION_PROMPT.format(
            role=role,
            profile_summary=profile_summary,
            first_step=json.dumps(first_step, indent=2)
        )
    else:
        # Get current plan step
        if current_step < len(interview_plan):
            current_plan_step = json.dumps(interview_plan[current_step], indent=2)
        else:
            current_plan_step = "Final wrap-up — ask if the candidate has any questions"

        # Get recent history (last 4 exchanges)
        recent = conversation_history[-8:] if len(conversation_history) > 8 else conversation_history
        recent_text = "\n".join([
            f"{'Interviewer' if msg.get('role') == 'interviewer' else 'Candidate'}: {msg.get('content', '')}"
            for msg in recent
            if isinstance(msg, dict) and 'content' in msg
        ])

        # Build instruction based on evaluation feedback
        if probe_deeper and evaluation:
            instruction = f"The candidate's last answer was weak (depth: {evaluation.get('depth', 'N/A')}). Ask a targeted follow-up to probe deeper on the same topic. Do NOT move to the next topic yet."
        elif evaluation.get("should_trigger_workspace", False):
            instruction = "Transition the candidate to a hands-on coding challenge. Explain that you'd like them to write some code."
        else:
            instruction = "Move naturally to the next topic in the plan. Briefly acknowledge the previous answer."

        last_eval_text = json.dumps(evaluation, indent=2) if evaluation else "No evaluation yet"

        prompt = QUESTION_GENERATION_PROMPT.format(
            role=role,
            profile_summary=profile_summary,
            current_plan_step=current_plan_step,
            recent_history=recent_text or "No conversation yet",
            last_evaluation=last_eval_text,
            instruction=instruction
        )

    try:
        question_text = await generate_text(
            prompt,
            system=INTERVIEWER_SYSTEM_PROMPT.format(role=role),
            temperature=0.7,
        )

        # Add to conversation history
        new_entry = {
            "role": "interviewer",
            "content": question_text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": current_step
        }

        return {
            "current_question": question_text,
            "conversation_history": [new_entry],
            "mode": "conversation",
            "probe_deeper": False  # Reset after use
        }

    except Exception as e:
        fallback = f"Could you tell me more about your experience with the topics relevant to this {role} position?"
        new_entry = {
            "role": "interviewer",
            "content": fallback,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": current_step,
            "_error": str(e)
        }
        return {
            "current_question": fallback,
            "conversation_history": [new_entry],
            "mode": "conversation",
            "probe_deeper": False
        }
