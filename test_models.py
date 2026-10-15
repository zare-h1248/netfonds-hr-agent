from dotenv import load_dotenv
from google import genai

load_dotenv("hr_agent/.env")
client = genai.Client()

candidates = [
    "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash",
    "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite",
    "gemini-flash-latest", "gemini-flash-lite-latest",
]

for name in candidates:
    try:
        client.models.generate_content(model=name, contents="Say OK")
        print(f"OK      {name}")
    except Exception as e:
        print(f"FAILED  {name}: {str(e)[:60]}")

try:
    client.models.embed_content(model="gemini-embedding-001", contents=["test"])
    print("OK      gemini-embedding-001 (embeddings)")
except Exception as e:
    print(f"FAILED  gemini-embedding-001: {str(e)[:60]}")