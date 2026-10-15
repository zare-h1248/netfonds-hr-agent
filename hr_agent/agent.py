import os
from pathlib import Path

from google.adk.agents import Agent
from pypdf import PdfReader

from .rag import search_documents
from google.genai import types


DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"

# "stuffing" = all documents in the prompt, "rag" = search with a tool
KNOWLEDGE_MODE = os.getenv("KNOWLEDGE_MODE", "stuffing").lower()
MODEL = os.getenv("MODEL", "gemini-3.5-flash-lite")


def load_documents() -> str:
    """Read all PDFs in docs/ and return their text, labelled by file name."""
    parts = []
    for pdf_path in sorted(DOCS_DIR.glob("*.pdf")):
        reader = PdfReader(pdf_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        parts.append(f"### Document: {pdf_path.name}\n{text}")
    return "\n\n".join(parts)


BASE_RULES = """
You are the HR onboarding assistant for new employees.

Rules:
- Answer ONLY based on the onboarding documents.
- Be as specific as possible: give concrete numbers, times, contacts.
- Name the document you used, e.g. (Source: 03_Working_Hours_Vacation_Sick_Leave.pdf).
- If the answer is not in the documents, say so honestly and refer the
  person to the HR team. Do not invent company rules.
- Answer in the language the user writes in (German or English).
"""

if KNOWLEDGE_MODE == "rag":
    SYSTEM_INSTRUCTION = BASE_RULES + """
- For EVERY question, first call the tool `search_documents` with the question
  (in English, because the documents are in English).
- Answer only from the passages the tool returns.
"""
    TOOLS = [search_documents]
else:
    SYSTEM_INSTRUCTION = BASE_RULES + f"\nOnboarding documents:\n{load_documents()}"
    TOOLS = []


def build_instruction(context) -> str:
    # Passing a function instead of a string stops ADK from treating
    # curly braces in the PDF text as template variables.
    return SYSTEM_INSTRUCTION



root_agent = Agent(
    name="hr_onboarding_agent",
    model=MODEL,
    description="Answers onboarding questions for new employees based on HR documents.",
    instruction=build_instruction,
    tools=TOOLS,
    generate_content_config=types.GenerateContentConfig(
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=5, initial_delay=2)
        )
    ),
)