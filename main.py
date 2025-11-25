"""
UX Interview Assistant - FastAPI Backend
Receives transcripts from Deepgram and generates interview questions using CreateAI (OpenAI models)
"""

import os
import time
from datetime import datetime
from typing import List, Dict, Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import google.generativeai as genai
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="UX Interview Assistant", version="2.0.0")

# CORS middleware to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize API keys
gemini_api_key = os.getenv("GEMINI_API_KEY")
createai_token = os.getenv("CREATEAI_TOKEN")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable is required")
if not createai_token:
    raise ValueError("CREATEAI_TOKEN environment variable is required")

genai.configure(api_key=gemini_api_key)

# CreateAI API configuration
CREATEAI_API_URL = "https://api-main-beta.aiml.asu.edu/query"

# In-memory transcript storage
transcript_history: List[Dict] = []
transcript_summary: str = ""  # Stores summarized older messages

# Configuration for summarization
MAX_MESSAGES_BEFORE_SUMMARY = 15
KEEP_RECENT_MESSAGES = 10


class TranscriptData(BaseModel):
    """Model for direct transcript data"""
    text: str
    speaker: Optional[str] = None
    timestamp: Optional[str] = None


def append_to_transcript(text: str, speaker: Optional[str] = None):
    """
    Append new transcript text to history
    """
    global transcript_history

    entry = {
        "text": text,
        "speaker": speaker or "Unknown",
        "timestamp": datetime.now().isoformat()
    }
    transcript_history.append(entry)


def summarize_transcript(messages_to_summarize: List[Dict]) -> str:
    """
    Use Gemini to summarize older transcript messages for context preservation
    Uses the UX Research Summarization system prompt
    """
    if not messages_to_summarize:
        return ""

    # Format messages for summarization
    transcript_text = "\n".join([
        f"[{msg.get('speaker', 'Unknown')}]: {msg['text']}"
        for msg in messages_to_summarize
    ])

    # UX Research Summarization System Prompt
    system_prompt = """🧠 System Prompt for Gemini API — UX Research Summarization

Role & Purpose:
You are a professional UX research summarizer assistant.
Your job is to compress earlier interview transcript segments into a concise, factual summary that preserves all critical UX research insights while removing filler conversation and repetitive statements.
This summary will be used by another AI model to continue analyzing the conversation and suggest follow-up questions — so every detail you keep must help that process.

Core Instruction:
Summarize the conversation up to the last 10 messages, ensuring that no key qualitative insight is lost.
Capture who the participant is, their behavior or persona type, stated goals, pain points, barriers, motivations, and attitudes.
Omit greetings, fillers, or irrelevant chit-chat.

Guidelines:

Keep the summary concise but information-dense (around 150–250 words).

Use neutral, research-style language — no interpretation or speculation.

Group related insights under clear bullet points or short paragraphs.

Include examples or quotes only if they illustrate an important finding.

Preserve nuances such as contradictions, confusion, or emotional tone if they indicate user frustration or satisfaction.

Do not remove or generalize pain points, workarounds, or adoption barriers — they are essential.

End with a short "Summary snapshot" section that highlights:

Key persona characteristics

Main frustrations or barriers

Desired outcomes or motivations"""

    full_prompt = f"""{system_prompt}

Transcript to summarize:
{transcript_text}

Provide a concise summary following the guidelines above."""

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            full_prompt,
            generation_config={
                'temperature': 0.7,
                'max_output_tokens': 500,
            }
        )
        return response.text.strip()

    except Exception as e:
        print(f"Error summarizing transcript: {e}")
        # Return a basic summary if Gemini fails
        return f"Previous conversation context: {len(messages_to_summarize)} messages discussed various topics."


def prepare_context_for_createai() -> str:
    """
    Prepare transcript context for CreateAI
    - If > 15 messages: summarize old messages, keep recent 10
    - Otherwise: use full transcript
    """
    global transcript_summary, transcript_history

    if len(transcript_history) > MAX_MESSAGES_BEFORE_SUMMARY:
        # Summarize old messages (all except last 10)
        messages_to_summarize = transcript_history[:-KEEP_RECENT_MESSAGES]
        transcript_summary = summarize_transcript(messages_to_summarize)

        # Keep only recent messages
        recent_messages = transcript_history[-KEEP_RECENT_MESSAGES:]
    else:
        # Use all messages without summarization
        recent_messages = transcript_history

    # Build context
    context_parts = []

    if transcript_summary:
        context_parts.append(f"[PREVIOUS CONTEXT SUMMARY]\n{transcript_summary}\n")

    if recent_messages:
        recent_text = "\n".join([
            f"[{msg['speaker']}]: {msg['text']}"
            for msg in recent_messages
        ])
        context_parts.append(f"[RECENT TRANSCRIPT]\n{recent_text}")

    return "\n\n".join(context_parts) if context_parts else "No transcript available yet."


def generate_questions_with_createai(context: str) -> List[str]:
    """
    Call CreateAI API to generate 2-3 interview questions using OpenAI gpt4o model
    Uses project settings configured in CreateAI UI
    """
    if not context or context == "No transcript available yet.":
        return [
            "What brings you here today?",
            "Can you tell me about your experience?",
            "What challenges have you faced?"
        ]

    try:
        headers = {
            "Authorization": f"Bearer {createai_token}",
            "Content-Type": "application/json"
        }

        payload = {
            "action": "query",
            "query": context,
            "model_provider": "openai",
            "model_name": "gpt4o",
            "request_source": "override_params"  # Use CreateAI project settings
        }

        response = requests.post(
            CREATEAI_API_URL,
            json=payload,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()

        data = response.json()

        # Parse response from CreateAI
        response_text = data.get("response", "")

        if not response_text:
            raise ValueError("Empty response from CreateAI")

        print(f"CreateAI response: {response_text}")  # Debug log

        # Try to parse JSON from the response (it might be wrapped in markdown code blocks)
        questions = []

        # Remove markdown code blocks if present
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()
        elif response_text.startswith("```"):
            response_text = response_text.replace("```", "").strip()

        # Try to parse as JSON
        try:
            import json
            parsed_json = json.loads(response_text)

            # Extract questions from JSON
            if isinstance(parsed_json, dict) and "questions" in parsed_json:
                questions = parsed_json["questions"]
            elif isinstance(parsed_json, list):
                questions = parsed_json
            else:
                raise ValueError("Unexpected JSON format")

        except (json.JSONDecodeError, ValueError) as json_error:
            print(f"Failed to parse as JSON: {json_error}")
            # Fall back to line-by-line parsing
            questions = [q.strip() for q in response_text.split("\n") if q.strip()]
            questions = [q.lstrip("1234567890.-) ").strip() for q in questions]
            # Remove common non-question lines
            questions = [q for q in questions if len(q) > 10 and not q.lower().startswith(("json", "{", "}", "[", "]"))]

        # Ensure we have 2-3 questions
        if len(questions) > 3:
            questions = questions[:3]
        elif len(questions) < 2:
            # Fallback questions if AI doesn't generate enough
            questions.extend([
                "Can you tell me more about that?",
                "What was that experience like for you?"
            ])
            questions = questions[:3]

        return questions[:3]

    except Exception as e:
        print(f"Error calling CreateAI API: {e}")
        # Return fallback questions on error
        return [
            "Can you tell me more about that?",
            "What was that experience like for you?",
            "How did that make you feel?"
        ]


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend HTML page"""
    try:
        with open("index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse("""
        <html>
            <head><title>UX Interview Assistant</title></head>
            <body>
                <h1>UX Interview Assistant</h1>
                <p>Please ensure index.html exists in the project directory.</p>
            </body>
        </html>
        """)


@app.post("/transcript")
async def receive_transcript(request: Request):
    """
    Receives transcripts from Deepgram real-time transcription
    """
    try:
        data = await request.json()

        # Handle Deepgram webhook format
        if "channel" in data and "alternatives" in data:
            # Deepgram real-time transcription webhook
            channel = data.get("channel", {})
            alternatives = channel.get("alternatives", [])
            if alternatives:
                transcript_text = alternatives[0].get("transcript", "")
                speaker = data.get("speaker", channel.get("speaker", "Participant"))
                if transcript_text:
                    append_to_transcript(transcript_text, speaker)
                    return {"status": "success", "message": "Deepgram transcript received"}

        # Handle Deepgram Results format
        elif "results" in data:
            results = data.get("results", {})
            if "channels" in results:
                for channel in results["channels"]:
                    for alternative in channel.get("alternatives", []):
                        transcript_text = alternative.get("transcript", "")
                        if transcript_text:
                            append_to_transcript(transcript_text, "Participant")
                return {"status": "success", "message": "Deepgram results received"}

        # Handle manual input / direct transcript data
        if "text" in data:
            text = data.get("text", "")
            speaker = data.get("speaker", "Participant")

            # Handle batch input (multiple lines)
            if isinstance(text, str) and "\n" in text:
                lines = [line.strip() for line in text.split("\n") if line.strip()]
                for line in lines:
                    # Try to detect speaker from line format
                    speaker_match = None
                    if ":" in line:
                        parts = line.split(":", 1)
                        if len(parts) == 2:
                            potential_speaker = parts[0].strip()
                            if len(potential_speaker) < 30:
                                speaker_match = potential_speaker
                                text_to_add = parts[1].strip()
                            else:
                                text_to_add = line
                        else:
                            text_to_add = line
                    elif line.startswith("[") and "]" in line:
                        bracket_end = line.index("]")
                        speaker_match = line[1:bracket_end]
                        text_to_add = line[bracket_end + 1:].strip()
                    else:
                        text_to_add = line

                    if text_to_add:
                        append_to_transcript(text_to_add, speaker_match or speaker)
            else:
                # Single text entry
                if text:
                    append_to_transcript(text, speaker)

            return {"status": "success", "message": "Transcript received"}

        return {"status": "received", "message": "Event processed"}

    except Exception as e:
        print(f"Error processing transcript: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/suggestions")
async def get_suggestions():
    """
    Returns 2-3 suggested interview questions based on current transcript
    Uses CreateAI with OpenAI gpt4o model
    Frontend polls this endpoint every few seconds
    """
    context = prepare_context_for_createai()
    questions = generate_questions_with_createai(context)

    return {
        "questions": questions,
        "timestamp": datetime.now().isoformat(),
        "transcript_length": len(transcript_history)
    }


@app.get("/transcript")
async def get_transcript():
    """
    Optional endpoint to view current transcript
    Useful for debugging
    """
    recent_entries = transcript_history[-20:] if transcript_history else []
    transcript_text = "\n".join([
        f"[{entry.get('speaker', 'Unknown')}]: {entry['text']}"
        for entry in recent_entries
    ])

    return {
        "transcript": transcript_text if transcript_text else "No transcript available yet.",
        "entries_count": len(transcript_history),
        "has_summary": bool(transcript_summary),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/deepgram-key")
async def get_deepgram_key():
    """
    Returns Deepgram API key for frontend WebSocket connection
    In production, use a more secure method (e.g., generate temporary tokens)
    """
    deepgram_key = os.getenv("DEEPGRAM_API_KEY")
    if not deepgram_key:
        return {
            "api_key": None,
            "message": "DEEPGRAM_API_KEY not set. Get a free API key at https://deepgram.com (60 hours/month free)"
        }
    return {
        "api_key": deepgram_key,
        "message": "Deepgram API key available"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
