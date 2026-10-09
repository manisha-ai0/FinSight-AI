# test.py
import os, time
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
for model in ["gemini-3.7-flash", "gemini-2.5-flash"]:
    t = time.time()
    try:
        r = client.models.generate_content(model=model, contents="Say hi")
        print(model, "OK", round(time.time() - t, 1), "s:", r.text[:50])
    except Exception as e:
        print(model, "FAILED", round(time.time() - t, 1), "s:", e)