#!/usr/bin/env python3
import os
import json
import time
import asyncio
import logging
import configparser
import sys
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential

import google.generativeai as genai

# ─── Configuration ───────────────────────────────────────────────────────────────

def load_config() -> Dict[str, Any]:
    """Load configuration from file."""
    config = configparser.ConfigParser()
    config.read("config.ini")
    return {
        "api_key": config["API_KEYS"]["paidAPIKey"],
        "transcribe_model": "gemini-2.0-flash",
        "analysis_model": "gemini-1.5-flash",
        "max_retries": 3,
        "batch_size": 5  # Number of files to process concurrently
    }

# Logging setup with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("processing.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Load configuration
config = load_config()
genai.configure(api_key=config["api_key"])

# Initialize the Gemini models
TRANSCRIBE_MODEL = config["transcribe_model"]
ANALYSIS_MODEL = config["analysis_model"]
ANALYSIS_CONFIG = {"response_mime_type": "application/json"}

# Your JSON-schema prompt (we'll append the transcript at runtime)
QA_PROMPT_TEMPLATE = """
You are an expert Quality Assurance Analyst for a law firm's customer service team.
Analyze the following call transcript and return your response strictly in this format:
{{
  "call_summary": "A brief, 3–4 sentence summary of the call including the reason for calling, main issues discussed, and any resolutions or next steps.",
  "qa_evaluation": [
    {{ "criterion": "Did the agent greet the caller warmly and professionally?", "score": "Yes/No", "justification": "..." }},
    {{ "criterion": "Did the agent confirm the caller's name and contact details?", "score": "Yes/No", "justification": "..." }},
    {{ "criterion": "Did the agent listen actively and avoid interrupting?", "score": "Yes/No", "justification": "..." }},
    {{ "criterion": "Did the agent ask appropriate questions to understand the issue?", "score": "Yes/No", "justification": "..." }},
    {{ "criterion": "Did the agent provide accurate and clear information?", "score": "Yes/No", "justification": "..." }},
    {{ "criterion": "Did the agent show empathy and concern for the caller's situation?", "score": "Yes/No", "justification": "..." }},
    {{ "criterion": "Did the agent avoid using filler words or inappropriate language?", "score": "Yes/No", "justification": "..." }},
    {{ "criterion": "Did the agent follow correct transfer/escalation procedures if needed?", "score": "Yes/No", "justification": "..." }},
    {{ "criterion": "Did the agent summarize the resolution or next steps before ending the call?", "score": "Yes/No", "justification": "..." }},
    {{ "criterion": "Did the agent close the call politely and professionally?", "score": "Yes/No", "justification": "..." }}
  ]
}}
---
Transcript:
{transcript}
"""

# ─── Helper Functions ────────────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
async def transcribe_audio(file_path: str) -> str:
    """Upload an audio file to Gemini and return the raw transcript text."""
    logging.info(f"Starting transcription for: {file_path}")
    try:
        with genai.upload_file(path=file_path) as audio:
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

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
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
        # Time transcription
        transcribe_start = time.time()
        transcript = await transcribe_audio(file_path)
        transcribe_time = time.time() - transcribe_start
        
        # Time analysis
        analysis_start = time.time()
        qa_json = await analyze_transcript(transcript)
        analysis_time = time.time() - analysis_start
        
        total_time = time.time() - start
        result = {
            "FileName": os.path.basename(file_path),
            "FilePath": file_path,
            "Success": True,
            "Transcription": transcript,
            "CallSummary": qa_json["call_summary"],
            "QA_Evaluation": qa_json["qa_evaluation"],
            "Timing": {
                "TranscriptionTime": round(transcribe_time, 2),
                "AnalysisTime": round(analysis_time, 2),
                "TotalTime": round(total_time, 2)
            }
        }
        logging.info(f"Successfully processed {file_path} in {total_time:.1f}s (Transcription: {transcribe_time:.1f}s, Analysis: {analysis_time:.1f}s)")
        return result
    except Exception as e:
        elapsed = time.time() - start
        logging.exception(f"Error processing {file_path} after {elapsed:.1f}s")
        return {
            "FileName": os.path.basename(file_path),
            "FilePath": file_path,
            "Success": False,
            "Error": str(e),
            "Timing": {
                "TotalTime": round(elapsed, 2)
            }
        }

async def process_batch(files: List[str]) -> List[Dict[str, Any]]:
    """Process a batch of files concurrently."""
    tasks = [process_file(p) for p in files]
    return await asyncio.gather(*tasks)

async def main(directory: str):
    total_start = time.time()
    # Gather all audio files
    audio_files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith((".mp3", ".wav"))
    ]
    logging.info(f"Found {len(audio_files)} audio files in {directory}")
    
    # Process in batches
    batch_size = config["batch_size"]
    results = []
    for i in range(0, len(audio_files), batch_size):
        batch = audio_files[i:i + batch_size]
        logging.info(f"Processing batch {i//batch_size + 1} of {(len(audio_files) + batch_size - 1)//batch_size}")
        batch_results = await process_batch(batch)
        results.extend(batch_results)

    # Calculate total processing time
    total_time = time.time() - total_start
    
    # Add summary statistics
    summary = {
        "TotalFiles": len(audio_files),
        "SuccessfulFiles": sum(1 for r in results if r["Success"]),
        "FailedFiles": sum(1 for r in results if not r["Success"]),
        "TotalProcessingTime": round(total_time, 2),
        "AverageTimePerFile": round(total_time / len(audio_files), 2) if audio_files else 0,
        "Results": results
    }

    # Save a master JSON with proper Unicode handling
    output_path = "output/all_calls.json"
    os.makedirs("output", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        json.dump(summary, fout, indent=2, ensure_ascii=False)
    logging.info(f"All results saved to {output_path}")
    logging.info(f"Total processing time: {total_time:.1f}s for {len(audio_files)} files (avg {total_time/len(audio_files):.1f}s per file)")

if __name__ == "__main__":
    asyncio.run(main(directory="test-transcription"))