"""
UX Interview Assistant - FastAPI Backend
Receives transcripts from Deepgram and generates interview questions using CreateAI (OpenAI models)
"""

import os
import time
import csv
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

# Freepik API configuration
FREEPIK_API_KEY = "FPSXeb5fa24989eb418083d3af38e92cfc4c"
FREEPIK_API_URL = "https://api.freepik.com/v1/resources"

# In-memory transcript storage
transcript_history: List[Dict] = []
transcript_summary: str = ""  # Stores summarized older messages

# In-memory storage for product info and generated questions
product_info: Dict = {}
generated_questions: List[Dict] = []
interview_qa_data: List[Dict] = []  # Stores questions asked during interview with answers

# Load interview database
def load_interview_database():
    """Load the 20 interviews from CSV database"""
    try:
        with open('customer_interviews.csv', 'r') as f:
            reader = csv.DictReader(f)
            return list(reader)
    except Exception as e:
        print(f"Error loading interview database: {e}")
        return []

interview_database = load_interview_database()

# Configuration for summarization
MAX_MESSAGES_BEFORE_SUMMARY = 15
KEEP_RECENT_MESSAGES = 10


class TranscriptData(BaseModel):
    """Model for direct transcript data"""
    text: str
    speaker: Optional[str] = None
    timestamp: Optional[str] = None


class ProductInput(BaseModel):
    """Model for product description input"""
    product_description: str
    research_goals: str


class QuestionUpdate(BaseModel):
    """Model for updating question list"""
    questions: List[Dict]


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
        # Use Gemini Flash latest model
        model = genai.GenerativeModel('models/gemini-flash-latest')
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


@app.post("/generate-questions")
async def generate_interview_questions(product_input: ProductInput):
    """
    Generate interview questions based on product description and research goals
    Uses Gemini API to create categorized questions
    """
    global product_info, generated_questions

    try:
        # Store product info
        product_info = {
            "product_description": product_input.product_description,
            "research_goals": product_input.research_goals,
            "timestamp": datetime.now().isoformat()
        }

        # Create prompt for Gemini
        prompt = f"""You are a UX research expert. Based on the product and research goals below, generate 12-15 interview questions organized into relevant categories.

Product Description:
{product_input.product_description}

Research Goals:
{product_input.research_goals}

Generate questions in the following JSON format:
{{
  "categories": [
    {{
      "name": "Category Name",
      "questions": ["Question 1?", "Question 2?", "Question 3?"]
    }}
  ]
}}

Create 3-4 categories relevant to the research goals (e.g., "User Background", "Pain Points & Needs", "Current Solutions", "Feature Validation", "Emotional Response", etc.).
Each category should have 3-5 questions.
Questions should be open-ended and designed for customer discovery interviews.
Return ONLY the JSON, no other text."""

        # Use Gemini Flash latest model
        model = genai.GenerativeModel('models/gemini-flash-latest')

        response = model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.8,
                'max_output_tokens': 2000,
            }
        )

        response_text = response.text.strip()

        # Clean up markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()
        elif response_text.startswith("```"):
            response_text = response_text.replace("```", "").strip()

        # Parse JSON response
        import json
        questions_data = json.loads(response_text)

        # Format questions with IDs
        formatted_categories = []
        question_id = 1
        for category in questions_data.get("categories", []):
            formatted_questions = []
            for q in category.get("questions", []):
                formatted_questions.append({
                    "id": question_id,
                    "text": q,
                    "selected": True
                })
                question_id += 1

            formatted_categories.append({
                "name": category.get("name", "General"),
                "questions": formatted_questions
            })

        generated_questions = formatted_categories

        return {
            "status": "success",
            "categories": formatted_categories,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        print(f"Error generating questions: {e}")
        import traceback
        traceback.print_exc()
        # Return fallback questions
        fallback_questions = [
            {
                "name": "User Background",
                "questions": [
                    {"id": 1, "text": "Can you tell me about your role and what you do day-to-day?", "selected": True},
                    {"id": 2, "text": "How long have you been in this role?", "selected": True},
                    {"id": 3, "text": "What tools or products do you currently use?", "selected": True}
                ]
            },
            {
                "name": "Pain Points & Needs",
                "questions": [
                    {"id": 4, "text": "What are the biggest challenges you face in your work?", "selected": True},
                    {"id": 5, "text": "Can you walk me through a recent frustrating experience?", "selected": True},
                    {"id": 6, "text": "What would make your workflow easier?", "selected": True}
                ]
            },
            {
                "name": "Current Solutions",
                "questions": [
                    {"id": 7, "text": "How do you currently solve this problem?", "selected": True},
                    {"id": 8, "text": "What do you like about your current solution?", "selected": True},
                    {"id": 9, "text": "What's missing from your current approach?", "selected": True}
                ]
            }
        ]
        generated_questions = fallback_questions
        return {
            "status": "success",
            "categories": fallback_questions,
            "timestamp": datetime.now().isoformat()
        }


@app.get("/questions")
async def get_generated_questions():
    """Get the currently generated questions"""
    return {
        "categories": generated_questions,
        "product_info": product_info,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/questions")
async def update_questions(update: QuestionUpdate):
    """Update the question list (when user removes questions)"""
    global generated_questions
    generated_questions = update.questions
    return {
        "status": "success",
        "message": "Questions updated",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/start-interview")
async def start_interview():
    """Reset transcript history and prepare for new interview"""
    global transcript_history, transcript_summary, interview_qa_data
    transcript_history = []
    transcript_summary = ""
    interview_qa_data = []

    return {
        "status": "success",
        "message": "Interview session started",
        "timestamp": datetime.now().isoformat()
    }


# Disabled - not needed for simplified live Q&A only version
# @app.post("/end-interview")
async def end_interview_disabled():
    """Process interview data and prepare summary with Q&A tagging"""
    global interview_qa_data, transcript_history

    try:
        # Analyze transcript and categorize responses
        transcript_text = "\n".join([
            f"[{entry.get('speaker', 'Unknown')}]: {entry['text']}"
            for entry in transcript_history
        ])

        # Get selected questions for reference
        selected_questions = []
        for category in generated_questions:
            for q in category.get("questions", []):
                if q.get("selected", True):
                    selected_questions.append({
                        "category": category.get("name"),
                        "question": q.get("text")
                    })

        # Use Gemini to analyze and tag responses
        analysis_prompt = f"""You are analyzing a UX research interview. Extract the key questions asked and the interviewee's answers, then categorize them.

Interview Transcript:
{transcript_text}

Reference Questions (these were prepared):
{json.dumps(selected_questions, indent=2)}

Analyze the transcript and return a JSON object with this structure:
{{
  "qa_pairs": [
    {{
      "question": "The actual question asked",
      "answer": "The interviewee's response (summarized if long)",
      "category": "Most relevant category from reference questions",
      "key_insights": ["insight 1", "insight 2"]
    }}
  ]
}}

Extract 8-12 main Q&A pairs from the interview. Return ONLY the JSON."""

        # Use Gemini Flash latest model
        model = genai.GenerativeModel('models/gemini-flash-latest')
        response = model.generate_content(
            analysis_prompt,
            generation_config={
                'temperature': 0.7,
                'max_output_tokens': 3000,
            }
        )

        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()
        elif response_text.startswith("```"):
            response_text = response_text.replace("```", "").strip()

        import json
        analysis_data = json.loads(response_text)
        interview_qa_data = analysis_data.get("qa_pairs", [])

        return {
            "status": "success",
            "qa_data": interview_qa_data,
            "transcript_length": len(transcript_history),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        print(f"Error processing interview: {e}")
        # Return basic Q&A extraction
        fallback_qa = []
        for i, entry in enumerate(transcript_history[-10:]):
            if entry.get('speaker') != 'Participant':
                continue
            fallback_qa.append({
                "question": f"Discussion point {i+1}",
                "answer": entry.get('text', ''),
                "category": "General",
                "key_insights": []
            })

        interview_qa_data = fallback_qa
        return {
            "status": "success",
            "qa_data": fallback_qa,
            "transcript_length": len(transcript_history),
            "timestamp": datetime.now().isoformat()
        }


def fetch_freepik_image(query: str, limit: int = 1) -> List[str]:
    """Fetch images from Freepik API"""
    try:
        headers = {
            "x-freepik-api-key": FREEPIK_API_KEY,
            "Content-Type": "application/json"
        }

        params = {
            "term": query,
            "limit": limit,
            "order": "latest"
        }

        response = requests.get(
            FREEPIK_API_URL,
            headers=headers,
            params=params,
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            images = []
            for item in data.get('data', []):
                # Correct path: item['image']['source']['url']
                if 'image' in item and 'source' in item['image'] and 'url' in item['image']['source']:
                    images.append(item['image']['source']['url'])
            print(f"✓ Fetched {len(images)} images from Freepik")
            return images
        else:
            print(f"Freepik API error: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        print(f"Error fetching Freepik images: {e}")
        return []


def analyze_interview_database():
    """Analyze all 20 interviews from the database"""
    if not interview_database:
        return None

    # Calculate aggregate statistics
    total_interviews = len(interview_database)

    # Industry breakdown
    industries = {}
    for interview in interview_database:
        ind = interview.get('industry', 'Unknown')
        industries[ind] = industries.get(ind, 0) + 1

    # Company size breakdown
    company_sizes = {}
    for interview in interview_database:
        size = interview.get('company_size', 'Unknown')
        company_sizes[size] = company_sizes.get(size, 0) + 1

    # Respondent types
    respondent_types = {}
    for interview in interview_database:
        rtype = interview.get('respondent_type', 'Unknown')
        respondent_types[rtype] = respondent_types.get(rtype, 0) + 1

    # Average sentiment
    sentiments = [float(i.get('sentiment_score', 0)) for i in interview_database if i.get('sentiment_score')]
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0

    # Recommendation rate
    recommendations = [int(i.get('would_recommend', 0)) for i in interview_database]
    recommendation_rate = (sum(recommendations) / len(recommendations) * 100) if recommendations else 0

    # Extract all key insights
    all_insights = []
    for interview in interview_database:
        insights_text = interview.get('key_insights', '')
        if insights_text:
            all_insights.append(insights_text)

    return {
        'total_interviews': total_interviews,
        'industries': industries,
        'company_sizes': company_sizes,
        'respondent_types': respondent_types,
        'avg_sentiment': round(avg_sentiment, 1),
        'recommendation_rate': round(recommendation_rate, 1),
        'all_insights': all_insights,
        'interviews': interview_database
    }


# Disabled - not needed for simplified live Q&A only version
# @app.post("/generate-report")
async def generate_report_disabled():
    """Generate comprehensive UX research report from 20 interview database with Freepik images"""
    try:
        # Analyze the interview database
        analysis = analyze_interview_database()

        if not analysis:
            return {
                "status": "error",
                "message": "Interview database not available",
                "timestamp": datetime.now().isoformat()
            }

        # Fetch relevant images from Freepik
        print("Fetching images from Freepik...")
        hero_images = fetch_freepik_image("user research interview analysis", limit=1)
        insight_images = fetch_freepik_image("data visualization charts", limit=1)
        persona_images = fetch_freepik_image("business professional person", limit=1)

        # Prepare comprehensive context for Gemini
        insights_summary = "\n\n".join(analysis['all_insights'][:10])  # First 10 insights

        report_prompt = f"""You are a UX research expert creating a comprehensive research report based on 20 customer interviews.

**Interview Database Statistics:**
- Total Interviews: {analysis['total_interviews']}
- Industries: {', '.join([f"{k}: {v}" for k, v in analysis['industries'].items()])}
- Company Sizes: {', '.join([f"{k}: {v}" for k, v in analysis['company_sizes'].items()])}
- Respondent Types: {', '.join([f"{k}: {v}" for k, v in analysis['respondent_types'].items()])}
- Average Sentiment Score: {analysis['avg_sentiment']}/100
- Would Recommend Rate: {analysis['recommendation_rate']}%

**Sample Key Insights from Interviews:**
{insights_summary}

Create a professional research report with these sections:

## Executive Summary
(3-4 sentences summarizing the overall findings from all 20 interviews)

## Methodology
- Total Participants: {analysis['total_interviews']}
- Interview Period: January 2024
- Average Duration: 35 minutes
- Industries Represented: {', '.join(analysis['industries'].keys())}

## Key Findings
(6-8 bullet points with the most critical insights across all interviews, cite specific numbers and patterns)

## User Segments & Personas
(Describe 2-3 main user personas based on the data, including their characteristics, goals, and challenges)

## Pain Points & Frustrations
(Organize by theme with frequency and severity, use actual data)

## Opportunities & Recommendations
(5-7 actionable recommendations based on the interview data, prioritized by impact)

## Sentiment Analysis
- Overall Sentiment: {analysis['avg_sentiment']}/100
- Recommendation Rate: {analysis['recommendation_rate']}%
(Provide interpretation of these metrics)

## Notable Quotes
(Include 3-4 powerful direct quotes from the interviews that represent key themes)

## Next Steps
(Suggest concrete actions for the product team)

Format as markdown. Use specific numbers and data points. Be professional and actionable."""

        # Generate report with Gemini
        model = genai.GenerativeModel('models/gemini-flash-latest')
        response = model.generate_content(
            report_prompt,
            generation_config={
                'temperature': 0.7,
                'max_output_tokens': 4000,
            }
        )

        report_text = response.text.strip()

        # Add header with stats
        header_stats = f"""---
**Research Report | {analysis['total_interviews']} Interviews | Generated {datetime.now().strftime('%B %Y')}**

📊 **Quick Stats:** Avg Sentiment: {analysis['avg_sentiment']}/100 | Recommendation Rate: {analysis['recommendation_rate']}%

---

"""
        report_text = header_stats + report_text

        # Add images to the report
        hero_url = hero_images[0] if hero_images else None
        insight_url = insight_images[0] if insight_images else None
        persona_url = persona_images[0] if persona_images else None

        # Insert hero image at the top
        if hero_url:
            report_text = f"![Research Overview]({hero_url})\n\n" + report_text

        # Insert insights image after Key Findings
        if insight_url and "## Key Findings" in report_text:
            parts = report_text.split("## User Segments", 1)
            if len(parts) == 2:
                report_text = parts[0] + f"\n\n![Data Insights]({insight_url})\n\n## User Segments" + parts[1]

        # Insert persona image after User Segments
        if persona_url and "## User Segments" in report_text:
            parts = report_text.split("## Pain Points", 1)
            if len(parts) == 2:
                report_text = parts[0] + f"\n\n![User Personas]({persona_url})\n\n## Pain Points" + parts[1]

        return {
            "status": "success",
            "report": report_text,
            "timestamp": datetime.now().isoformat(),
            "database_stats": {
                "total_interviews": analysis['total_interviews'],
                "avg_sentiment": analysis['avg_sentiment'],
                "recommendation_rate": analysis['recommendation_rate']
            },
            "images": {
                "hero": hero_url,
                "insights": insight_url,
                "personas": persona_url
            }
        }

    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }


# Disabled - not needed for simplified live Q&A only version
# @app.get("/database-stats")
async def get_database_stats_disabled():
    """Get statistics about the interview database"""
    analysis = analyze_interview_database()

    if not analysis:
        return {
            "status": "error",
            "message": "Database not loaded"
        }

    return {
        "status": "success",
        "statistics": {
            "total_interviews": analysis['total_interviews'],
            "industries": analysis['industries'],
            "company_sizes": analysis['company_sizes'],
            "respondent_types": analysis['respondent_types'],
            "avg_sentiment": analysis['avg_sentiment'],
            "recommendation_rate": analysis['recommendation_rate']
        },
        "sample_insights": analysis['all_insights'][:3]
    }


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
