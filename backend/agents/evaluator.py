"""
Evaluation Agent.
Scores each candidate response on multiple dimensions.
Uses chain-of-thought reasoning before scoring to avoid fake/lazy scoring.
Updates cumulative scores and detects strengths/weaknesses dynamically.
"""

import json
from google import genai
from backend.config import settings

client = genai.Client(api_key=settings.gemini_api_key)

EVALUATION_PROMPT = """You are an expert technical interview evaluator. Analyze the candidate's response thoroughly.

## Context
**Role**: {role}
**Current Topic**: {current_topic}
**Question Asked**: {question}
**Candidate's Response**: {response}

**Candidate Profile** (for calibration):
- Experience Level: {experience_level}
- Relevant Skills: {relevant_skills}

**Previous Cumulative Scores**:
{cumulative_scores}

## Evaluation Instructions:
1. FIRST, think step-by-step about the response:
   - What did the candidate get right?
   - What did they get wrong or miss?
   - How deep was their understanding?
   - Did they communicate clearly?
   - How confident did they seem?

2. THEN score on these dimensions (0.0 to 1.0):
   - **correctness**: factual accuracy and technical correctness
   - **depth**: did they go beyond surface level? Did they show WHY, not just WHAT?
   - **clarity**: was the answer well-structured and easy to follow?
   - **confidence**: did they seem assured or uncertain? (inferred from language)

3. Decide on next actions:
   - **should_probe_deeper**: true if the answer was weak/vague and the topic is important
   - **should_trigger_workspace**: true if this topic would benefit from a hands-on coding test

## STRICT Scoring Guidelines:
- 0.0-0.2: Completely wrong or no answer
- 0.2-0.4: Major misconceptions, very surface level
- 0.4-0.6: Partially correct, some understanding
- 0.6-0.8: Mostly correct with reasonable depth
- 0.8-1.0: Excellent, expert-level response

DO NOT inflate scores. A mediocre answer should score 0.4-0.5, not 0.7.

## Output Format (JSON only):
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
        response = await client.aio.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
            )
        )

        eval_text = response.text.strip()
        if eval_text.startswith("```"):
            eval_text = eval_text.split("\n", 1)[1]
            if eval_text.endswith("```"):
                eval_text = eval_text[:-3]
            eval_text = eval_text.strip()

        evaluation = json.loads(eval_text)

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
