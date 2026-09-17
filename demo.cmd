 set SPEECH_TO_SPEECH_URL=ws://127.0.0.1:8765/v1/realtime
 set SERPER_API_KEY=llama
 set STARTUP_GREETING="How can I help?"
 uv run uvicorn --app-dir demo server:app --reload --port 7860 --host 0.0.0.0
