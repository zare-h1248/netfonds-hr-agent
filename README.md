# HR Onboarding Agent

Chat agent for HR onboarding questions, built with Google ADK and Gemini.
It answers from the PDFs in docs/ (sample data, fictional company).

KNOWLEDGE_MODE in .env switches between "stuffing" (all docs in the prompt)
and "rag" (search tool with embeddings).

## Setup
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy .env.example to hr_agent/.env and add your Gemini API key.

## Run
```
adk web
```