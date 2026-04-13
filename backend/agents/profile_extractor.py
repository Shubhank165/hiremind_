"""
Profile Extraction Agent.
Parses resume text and role to produce a structured candidate profile.
Uses chain-of-thought prompting to infer skill depth, not just keywords.
"""

import base64
import json

import fitz
from google import genai

from backend.config import settings
from backend.utils.gemini_fast import fast_config
from backend.utils.llm import generate_json
from backend.utils.resume_parser import parse_resume

gemini_client = genai.Client(api_key=settings.gemini_api_key) if settings.gemini_api_key else None

PROFILE_EXTRACTION_PROMPT = """You are an expert technical recruiter. Extract a structured candidate profile.

Requirements:
- Infer skill depth and evidence from projects and experience.
- Identify gaps for the target role.
- Be concise.

Target role: {role}

Resume content:
{resume_text}

Return JSON only (no markdown) with this schema:
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


def _pdf_to_images(pdf_bytes: bytes, max_pages: int = 2, dpi: int = 150) -> list[bytes]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images: list[bytes] = []
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB)
        images.append(pix.tobytes("png"))
    doc.close()
    return images


def _strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    return cleaned.strip()


def _extract_json_object(text: str) -> str:
    cleaned = _strip_code_fence(text)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return cleaned
    return cleaned[start:end + 1]


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

    if settings.fast_profile_mode:
        summary = f"Candidate applied for {role}. Fast profile mode: resume parsed lightly."
        return {
            "profile": {
                "skills": [],
                "experience_level": "unknown",
                "years_experience": "unknown",
                "projects": [],
                "education": {},
                "strengths": [],
                "weaknesses": ["Fast profile mode enabled"],
                "confidence_scores": {
                    "overall_fit": 0.3,
                    "technical_depth": 0.3,
                    "experience_match": 0.3,
                    "project_quality": 0.3
                },
                "key_topics_to_probe": ["core fundamentals", "project details"],
                "summary": summary
            },
            "mode": "planning"
        }

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

    try:
        resume_bytes = base64.b64decode(resume_data)

        if "pdf" in resume_mime and gemini_client is not None:
            images = _pdf_to_images(resume_bytes)
            parts = [{"text": PROFILE_EXTRACTION_PROMPT.format(role=role, resume_text="[See images]") }]
            for img in images:
                parts.append(genai.types.Part.from_bytes(data=img, mime_type="image/png"))

            response = await gemini_client.aio.models.generate_content(
                model=settings.gemini_model,
                contents=[{"role": "user", "parts": parts}],
                config=fast_config(
                    temperature=0.3,
                    response_mime_type="application/json",
                    max_output_tokens=max(settings.gemini_json_max_tokens, 3000),
                ),
            )
            profile = json.loads(_extract_json_object(response.text or ""))
        else:
            resume_filename = "resume.pdf" if "pdf" in resume_mime else "resume.txt"
            resume_text = parse_resume(resume_bytes, resume_filename)

            prompt = PROFILE_EXTRACTION_PROMPT.format(
                role=role,
                resume_text=resume_text or "[Empty resume text]",
            )

            profile = await generate_json(
                prompt,
                temperature=0.3,
            )

        return {
            "profile": profile,
            "mode": "planning"
        }

    except json.JSONDecodeError:
        # Fallback: extract what we can
        raw_response = ""
        try:
            raw_response = response.text if response else ""
        except Exception:
            raw_response = ""
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
                "_raw_response": raw_response
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
