#!/usr/bin/env python3
import os
import json
import time
import asyncio
import logging
import configparser
import sys

import google.generativeai as genai

# ─── Configuration ───────────────────────────────────────────────────────────────

# Logging setup with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("processing.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Load Gemini API key
config = configparser.ConfigParser()
config.read("config.ini")
API_KEY = config["API_KEYS"]["paidAPIKey"]
genai.configure(api_key=API_KEY)

# Initialize the Gemini models
TRANSCRIBE_MODEL = "gemini-2.0-flash"      # audio understanding model
ANALYSIS_MODEL   = "gemini-1.5-flash"      # text model for QA JSON
ANALYSIS_CONFIG  = {"response_mime_type": "application/json"}

# Your JSON-schema prompt (we'll append the transcript at runtime)
# QA_PROMPT_TEMPLATE = """
# You are an expert Quality Assurance Analyst for a law firm's customer service team.
# Analyze the following call transcript and return your response strictly in this format:
# {{
#   "call_summary": "A brief, 3–4 sentence summary of the call including the reason for calling, main issues discussed, and any resolutions or next steps.",
#   "qa_evaluation": [
#     {{ "criterion": "Did the agent greet the caller warmly and professionally?", "score": "Yes/No", "justification": "..." }},
#     {{ "criterion": "Did the agent confirm the caller's name and contact details?", "score": "Yes/No", "justification": "..." }},
#     {{ "criterion": "Did the agent listen actively and avoid interrupting?", "score": "Yes/No", "justification": "..." }},
#     {{ "criterion": "Did the agent ask appropriate questions to understand the issue?", "score": "Yes/No", "justification": "..." }},
#     {{ "criterion": "Did the agent provide accurate and clear information?", "score": "Yes/No", "justification": "..." }},
#     {{ "criterion": "Did the agent show empathy and concern for the caller's situation?", "score": "Yes/No", "justification": "..." }},
#     {{ "criterion": "Did the agent avoid using filler words or inappropriate language?", "score": "Yes/No", "justification": "..." }},
#     {{ "criterion": "Did the agent follow correct transfer/escalation procedures if needed?", "score": "Yes/No", "justification": "..." }},
#     {{ "criterion": "Did the agent summarize the resolution or next steps before ending the call?", "score": "Yes/No", "justification": "..." }},
#     {{ "criterion": "Did the agent close the call politely and professionally?", "score": "Yes/No", "justification": "..." }}
#   ]
# }}
# ---
# Transcript:
# {transcript}
# """
QA_PROMPT_TEMPLATE = """
You are a Quality Assurance Analyst evaluating intake calls for a law firm. Analyze the following call transcript and respond strictly in this JSON format:
{{
  "call_summary": "A brief 3–4 sentence summary including the reason for the call, key discussion points, and any resolutions or next steps.",
  "qa_evaluation": [[
    {{ "criterion": "Did the intake specialist greet with a warm, enthusiastic tone and slow pace?", "score": "Yes/No", "justification": "Explain why." }},
    {{ "criterion": "Did the intake specialist ask a personalization question like 'What made you call today?'", "score": "Yes/No", "justification": "Explain why." }},
    {{ "criterion": "Was empathy demonstrated consistently throughout the call?", "score": "Yes/No", "justification": "Explain why." }},
    {{ "criterion": "Did the intake specialist confidently control the call flow?", "score": "Yes/No", "justification": "Explain why." }},
    {{ "criterion": "Were unnecessary interruptions avoided?", "score": "Yes/No", "justification": "Explain why." }},
    {{ "criterion": "Was the caller’s name used during the conversation?", "score": "Yes/No", "justification": "Explain why." },
    {{ "criterion": "Was the caller’s phone number collected early in the call?", "score": "Yes/No", "justification": "Explain why." }},
    {{ "criterion": "Was the referral source asked at the end (e.g., 'Who referred you to us?')", "score": "Yes/No", "justification": "Explain why." }},
    {{ "criterion": "Did the agent advocate for the law firm (e.g., 'You’ve called the right place')?", "score": "Yes/No", "justification": "Explain why." }},
    {{ "criterion": "Did the agent close the call by thanking the caller (e.g., 'Thank you for calling, have a nice day')", "score": "Yes/No", "justification": "Explain why." }}
  ]]
}}
Transcript:
 ---
{transcript}
"""
# ─── Helper Functions ────────────────────────────────────────────────────────────

async def transcribe_audio(file_path: str) -> str:
    """Upload an audio file to Gemini and return the raw transcript text."""
    logging.info(f"Starting transcription for: {file_path}")
    audio = None
    try:
        audio = genai.upload_file(path=file_path)
        if not audio:
            raise RuntimeError(f"Upload failed for {file_path}")

        # Ask Gemini to return only the verbatim transcript
        response = genai.GenerativeModel(TRANSCRIBE_MODEL).generate_content([
            "Please provide only the verbatim transcription of this call, without timestamps or metadata.",
            audio
        ])
        transcript = response.text.strip()
        word_count = len(transcript.split())
        logging.info(f"Successfully transcribed {file_path} ({word_count} words)")
        return transcript
    except Exception as e:
        logging.error(f"Error during transcription of {file_path}: {str(e)}")
        raise
    finally:
        if audio:
            try:
                genai.delete_file(audio.name)
                logging.debug(f"Cleaned up uploaded file: {audio.name}")
            except Exception as e:
                logging.warning(f"Failed to clean up uploaded file {audio.name}: {str(e)}")

async def analyze_transcript(transcript: str) -> dict:
    """Feed the transcript into Gemini's text model and parse the JSON QA output."""
    logging.info("Starting transcript analysis")
    try:
        prompt = QA_PROMPT_TEMPLATE.format(transcript=transcript)
        model = genai.GenerativeModel(ANALYSIS_MODEL, generation_config=ANALYSIS_CONFIG)
        response = model.generate_content([prompt])
        result = json.loads(response.text)
        logging.info("Successfully analyzed transcript")
        return result
    except Exception as e:
        logging.error(f"Error during transcript analysis: {str(e)}")
        raise

async def process_file(file_path: str) -> dict:
    """Full pipeline: transcribe, analyze, collate results."""
    start = time.time()
    logging.info(f"Starting processing pipeline for: {file_path}")
    try:
        transcript = await transcribe_audio(file_path)
        qa_json = await analyze_transcript(transcript)
        result = {
            "FileName": os.path.basename(file_path),
            "FilePath": file_path,
            "Success": True,
            "CallSummary": qa_json["call_summary"],
            "QA_Evaluation": qa_json["qa_evaluation"]
        }
        elapsed = time.time() - start
        logging.info(f"Successfully processed {file_path} in {elapsed:.1f}s")
        return result
    except Exception as e:
        elapsed = time.time() - start
        logging.exception(f"Error processing {file_path} after {elapsed:.1f}s")
        return {
            "FileName": os.path.basename(file_path),
            "FilePath": file_path,
            "Success": False,
            "Error": str(e)
        }

async def main(directory: str):
    # Gather all audio files
    audio_files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith((".mp3", ".wav"))
    ]
    logging.info(f"Found {len(audio_files)} audio files in {directory}")
    
    # Process in parallel
    tasks = [process_file(p) for p in audio_files]
    results = await asyncio.gather(*tasks)

    # Save a master JSON with proper Unicode handling
    output_path = "output/all_calls.json"
    os.makedirs("output", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        json.dump(results, fout, indent=2, ensure_ascii=False)
    logging.info(f"All results saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(main(directory="test-transcription"))
