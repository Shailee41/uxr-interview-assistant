# UXR Interview Assistant

An AI-powered interview assistant that provides real-time transcription and generates contextual follow-up questions for UX research interviews.

## Features

- **Real-time Audio Transcription**: Uses Deepgram for live speech-to-text
- **AI-Powered Question Generation**: Generates relevant follow-up questions using OpenAI (via CreateAI)
- **Transcript Summarization**: Automatically summarizes long conversations using Google Gemini
- **Beautiful UI**: Soft, minimal design with glassmorphism aesthetics

## Tech Stack

- **Backend**: FastAPI + Python
- **AI Models**:
  - OpenAI GPT-4 (via CreateAI) for question generation
  - Google Gemini for transcript summarization
  - Deepgram for real-time transcription
- **Frontend**: Vanilla JavaScript with modern CSS

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file (copy from `env.template`):
```bash
cp env.template .env
```

3. Add your API keys to `.env`:
```
GEMINI_API_KEY=your_gemini_api_key_here
CREATEAI_TOKEN=your_createai_token_here
DEEPGRAM_API_KEY=your_deepgram_api_key_here
```

4. Run the server:
```bash
python3 main.py
```

5. Open [http://localhost:8000](http://localhost:8000)

## Deploy to Render.com (Free)

### Step 1: Push to GitHub

1. Initialize git repository (if not already done):
```bash
cd "/Users/shaileeshah/UXR interviewer"
git init
git add .
git commit -m "Initial commit"
```

2. Create a new repository on GitHub:
   - Go to [github.com/new](https://github.com/new)
   - Name it: `uxr-interview-assistant`
   - Don't initialize with README (we already have one)
   - Click "Create repository"

3. Push your code:
```bash
git remote add origin https://github.com/YOUR_USERNAME/uxr-interview-assistant.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy on Render

1. Sign up at [render.com](https://render.com) (free, no credit card required)

2. Click "New +" → "Web Service"

3. Connect your GitHub repository:
   - Click "Connect account" to link GitHub
   - Select `uxr-interview-assistant`

4. Configure the service (Render auto-detects from `render.yaml`):
   - Name: `uxr-interview-assistant` (or any name you want)
   - Branch: `main`
   - Render will use settings from `render.yaml`

5. Add environment variables:
   - Click "Environment" tab
   - Add these three variables:
     - `GEMINI_API_KEY`: Your Gemini API key
     - `CREATEAI_TOKEN`: Your CreateAI token
     - `DEEPGRAM_API_KEY`: Your Deepgram API key

6. Click "Create Web Service"

7. Wait 2-3 minutes for deployment

8. Your app will be live at: `https://uxr-interview-assistant.onrender.com`

### Free Tier Limitations

- App sleeps after 15 minutes of inactivity
- Wakes up automatically when someone visits (takes ~30 seconds)
- 750 hours/month of runtime (plenty for demos)

## API Keys (Free Tiers)

### Gemini API
- Get free key: [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
- Free tier: 60 requests/minute

### Deepgram API
- Get free key: [https://deepgram.com](https://deepgram.com)
- Free tier: 60 hours/month of transcription

### CreateAI Token
- Contact: Ayat Sweid (Ayat.Sweid@asu.edu) or Paul Alvarado (Paul.Alvarado.1@asu.edu)

## Usage

1. Click "START RECORDING" to begin recording audio
2. Speak naturally - the transcript will appear in real-time on the right
3. Click "GENERATE QUESTIONS" to get AI-powered follow-up questions
4. Click "STOP RECORDING" when done

## Project Structure

```
.
├── main.py              # FastAPI backend
├── index.html           # Frontend UI
├── requirements.txt     # Python dependencies
├── render.yaml          # Render deployment config
├── .env                 # Environment variables (not in git)
├── env.template         # Template for .env
└── README.md            # This file
```

## License

MIT
