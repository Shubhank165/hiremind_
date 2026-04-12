"""
Profile Extraction Agent.
Parses resume text and role to produce a structured candidate profile.
Uses chain-of-thought prompting to infer skill depth, not just keywords.
"""

import json
from google import genai
from backend.config import settings

client = genai.Client(api_key=settings.gemini_api_key)

PROFILE_EXTRACTION_PROMPT = """You are an expert technical recruiter and talent analyst. Your job is to analyze a candidate's resume and extract a deep, structured profile.

**IMPORTANT**: Do NOT just extract keywords. You must INFER:
- Skill depth (beginner vs. practitioner vs. expert) based on project complexity, years of use, and context
- Confidence scores reflecting how certain you are about each skill assessment
- Gaps and weaknesses based on what's MISSING for the target role

## Target Role: {role}

## Resume Text:
{resume_text}

## Instructions:
1. First, think step-by-step about what this resume reveals about the candidate
2. Consider the target role requirements and identify alignments and gaps
3. Produce the structured profile below

## Output Format (JSON only, no markdown):
{{
    "skills": [
        {{
            "name": "skill name",
            "level": "beginner|intermediate|advanced|expert",
            "confidence": 0.0-1.0,
            "evidence": "brief note on why this level"
        }}
    ],
    "experience_level": "junior|mid|senior|staff",
    "years_experience": "estimated total years",
    "projects": [
        {{
            "name": "project name",
            "description": "what it does",
            "technologies": ["tech1", "tech2"],
            "impact": "business/technical impact",
            "complexity": "low|medium|high"
        }}
    ],
    "education": {{
        "degree": "degree name",
        "institution": "school name",
        "relevance": "how relevant to target role"
    }},
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["gap1", "gap2"],
    "confidence_scores": {{
        "overall_fit": 0.0-1.0,
        "technical_depth": 0.0-1.0,
        "experience_match": 0.0-1.0,
        "project_quality": 0.0-1.0
    }},
    "key_topics_to_probe": ["topic1", "topic2"],
    "summary": "2-3 sentence candidate summary"
}}
"""


async def extract_profile(state: dict) -> dict:
    """
    LangGraph node: Extract structured profile from resume.

    Args:
        state: Current interview state with 'resume_text' and 'role'

    Returns:
        State update with 'profile' and 'mode' fields
    """
    resume_data = state.get("resume_data", "")
    resume_mime = state.get("resume_mime_type", "text/plain")
    role = state.get("role", "Software Engineer")

    if not resume_data:
        # No resume provided — create minimal profile from role alone
        return {
            "profile": {
                "skills": [],
                "experience_level": "unknown",
                "years_experience": "unknown",
                "projects": [],
                "education": {},
                "strengths": [],
                "weaknesses": ["No resume provided — full assessment needed"],
                "confidence_scores": {
                    "overall_fit": 0.0,
                    "technical_depth": 0.0,
                    "experience_match": 0.0,
                    "project_quality": 0.0
                },
                "key_topics_to_probe": ["general technical ability", "role-specific knowledge"],
                "summary": f"No resume provided. Candidate applied for {role}. Full assessment required during interview."
            },
            "mode": "planning"
        }

    prompt = PROFILE_EXTRACTION_PROMPT.format(
        role=role,
        resume_text="[See attached Document]"
    )

    import base64
    document_part = genai.types.Part.from_bytes(
        data=base64.b64decode(resume_data),
        mime_type=resume_mime
    )

    try:
        response = await client.aio.models.generate_content(
            model=settings.gemini_model,
            contents=[document_part, prompt],
            config=genai.types.GenerateContentConfig(
                temperature=0.3,
                response_mime_type="application/json",
            )
        )

        profile_text = response.text.strip()
        # Clean potential markdown wrappers
        if profile_text.startswith("```"):
            profile_text = profile_text.split("\n", 1)[1]
            if profile_text.endswith("```"):
                profile_text = profile_text[:-3]
            profile_text = profile_text.strip()

        profile = json.loads(profile_text)

        return {
            "profile": profile,
            "mode": "planning"
        }

    except json.JSONDecodeError:
        # Fallback: extract what we can
        return {
            "profile": {
                "skills": [],
                "experience_level": "unknown",
                "years_experience": "unknown",
                "projects": [],
                "education": {},
                "strengths": [],
                "weaknesses": ["Profile extraction encountered parsing issues"],
                "confidence_scores": {
                    "overall_fit": 0.3,
                    "technical_depth": 0.3,
                    "experience_match": 0.3,
                    "project_quality": 0.3
                },
                "key_topics_to_probe": ["general technical ability"],
                "summary": f"Candidate applied for {role}. Automated profile extraction had issues — manual assessment recommended.",
                "_raw_response": response.text if response else ""
            },
            "mode": "planning"
        }
    except Exception as e:
        return {
            "profile": {
                "skills": [],
                "experience_level": "unknown",
                "strengths": [],
                "weaknesses": [f"Profile extraction failed: {str(e)}"],
                "confidence_scores": {},
                "key_topics_to_probe": ["general assessment"],
                "summary": f"Error during profile extraction for {role} role."
            },
            "mode": "planning"
        }
