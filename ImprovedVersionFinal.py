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

from google import genai
from google.genai import types

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
        "batch_size": 10  # Number of files to process concurrently
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
# Create a single client for the new SDK
client = genai.Client(api_key=config["api_key"])

# Initialize the Gemini models
TRANSCRIBE_MODEL = config["transcribe_model"]
ANALYSIS_MODEL = config["analysis_model"]

# Your JSON-schema prompt (we'll append the transcript at runtime)
QA_PROMPT_TEMPLATE = """
You are an expert Quality Assurance Analyst for a law firm's customer service team.
Analyze the following call transcript and return your response strictly in this format:
{{
  "call_summary": "A brief, 3–4 sentence summary of the call including the reason for calling, main issues discussed, and any resolutions or next steps. It should also include the agent's name and the caller's name.",
  "qa_evaluation": 
  [
  {{ "criterion": "Was the call opened with a friendly and professional tone?", "score": "Yes/No/NA", "justification": "..." }},
  {{ "criterion": "Did the agent ask appropriate questions to understand the issue?", "score": "Yes/No/NA", "justification": "..." }},
  {{ "criterion": "Was empathy demonstrated throughout the interaction?", "score": "Yes/No/NA", "justification": "..." }},
  {{ "criterion": "Did the agent maintain control and guide the call effectively?", "score": "Yes/No/NA", "justification": "..." }},
  {{ "criterion": "Did the agent avoid interrupting the caller unnecessarily?", "score": "Yes/No/NA", "justification": "..." }},
  {{ "criterion": "Did the agent personalize the interaction using the caller's name?", "score": "Yes/No/NA", "justification": "..." }},
  {{ "criterion": "Was important contact information (e.g., phone number) collected?", "score": "Yes/No/NA", "justification": "..." }},
  {{ "criterion": "Did the agent ask how the caller heard about the service (if it was a new inquiry)?", "score": "Yes/No/NA", "justification": "Explain why and mark N/A if it was a follow-up or existing case" }},
  {{ "criterion": "Did the agent express confidence in the service or organization?", "score": "Yes/No/NA", "justification": "..." }},
  {{ "criterion": "Was the call closed with a courteous and appreciative message?", "score": "Yes/No/NA", "justification": "..." }}
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
        # Open and read the file
        with open(file_path, 'rb') as audio_file:
            audio_data = audio_file.read()
        # Use the new SDK async client for transcription
        response = await client.aio.models.generate_content(
            model=TRANSCRIBE_MODEL,
            contents=[
                "Please provide only the verbatim transcription of this call, without timestamps or metadata.",
                types.Part.from_bytes(
                    data=audio_data,
                    mime_type="audio/wav" if file_path.lower().endswith('.wav') else "audio/mpeg"
                )
            ]
        )
        transcript = (response.text or "").strip()
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
        response = await client.aio.models.generate_content(
            model=ANALYSIS_MODEL,
            contents=[prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        if not response.text:
            raise ValueError("No response text from analysis model")
        result = json.loads(response.text or "{}")
        logging.info("Successfully analyzed transcript")
        return result
    except Exception as e:
        logging.error(f"Error during transcript analysis: {str(e)}")
        raise

async def process_file(file_path: str) -> dict:
    start = time.time()
    logging.info(f"Starting processing pipeline for: {file_path}")
    try:
        transcript = await transcribe_audio(file_path)
        await asyncio.sleep(2)  # 🕒 Throttle after transcription

        qa_json = await analyze_transcript(transcript)
        await asyncio.sleep(2)  # 🕒 Throttle after analysis

        total_time = time.time() - start
        return {
            "FileName": os.path.basename(file_path),
            "FilePath": file_path,
            "Success": True,
            "Transcription": transcript,
            "CallSummary": qa_json["call_summary"],
            "QA_Evaluation": qa_json["qa_evaluation"],
        }
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

# async def process_file(file_path: str) -> dict:
#     """Full pipeline: transcribe, analyze, collate results."""
#     start = time.time()
#     logging.info(f"Starting processing pipeline for: {file_path}")
#     try:
#         # Time transcription
#         transcribe_start = time.time()
#         transcript = await transcribe_audio(file_path)
#         transcribe_time = time.time() - transcribe_start
        
#         # Time analysis
#         analysis_start = time.time()
#         qa_json = await analyze_transcript(transcript)
#         analysis_time = time.time() - analysis_start
        
#         total_time = time.time() - start
#         result = {
#             "FileName": os.path.basename(file_path),
#             "FilePath": file_path,
#             "Success": True,
#             "Transcription": transcript,
#             "CallSummary": qa_json["call_summary"],
#             "QA_Evaluation": qa_json["qa_evaluation"],
#             # "Timing": {
#             #     "TranscriptionTime": round(transcribe_time, 2),
#             #     "AnalysisTime": round(analysis_time, 2),
#             #     "TotalTime": round(total_time, 2)
#             # }
#         }
#         logging.info(f"Successfully processed {file_path} in {total_time:.1f}s (Transcription: {transcribe_time:.1f}s, Analysis: {analysis_time:.1f}s)")
#         return result
#     except Exception as e:
#         elapsed = time.time() - start
#         logging.exception(f"Error processing {file_path} after {elapsed:.1f}s")
#         return {
#             "FileName": os.path.basename(file_path),
#             "FilePath": file_path,
#             "Success": False,
#             "Error": str(e),
#             "Timing": {
#                 "TotalTime": round(elapsed, 2)
#             }
#         }

async def process_batch(files: List[str]) -> List[Dict[str, Any]]:
    results = []
    for file in files:
        result = await process_file(file)
        results.append(result)
        await asyncio.sleep(2)  # 🕒 Space between file-level operations
    return results

# async def process_batch(files: List[str]) -> List[Dict[str, Any]]:
#     """Process a batch of files concurrently."""
#     results = []
#     for file in files:
#         result = await process_file(file)
#         results.append(result)
#         await asyncio.sleep(1.5)  # Prevent quota hit
#     return results
    
    # tasks = [process_file(p) for p in files]
    # results = []
    # for task in tasks:
    #     result = await task
    #     results.append(result)
    #     await asyncio.sleep(1.5)  # delay between each file to avoid 429
    # return results
    #return await asyncio.gather(*tasks)

async def main(directory: str):
    total_start = time.time()
    
    # Get all subdirectories that contain audio files
    subdirs = []
    for root, dirs, files in os.walk(directory):
        if any(f.lower().endswith((".mp3", ".wav")) for f in files):
            subdirs.append(root)
    
    if not subdirs:
        logging.error(f"No audio files found in directory or its subdirectories: {directory}")
        return
    
    # Process each subdirectory
    for subdir in subdirs:
        all_results = []
        total_files = 0
        
        # Gather audio files for this subdirectory
        audio_files = [
            os.path.join(subdir, f)
            for f in os.listdir(subdir)
            if f.lower().endswith((".mp3", ".wav"))
        ]
        
        if not audio_files:
            continue
            
        total_files += len(audio_files)
        logging.info(f"Found {len(audio_files)} audio files in {subdir}")
        
        # Process in batches
        batch_size = config["batch_size"]
        for i in range(0, len(audio_files), batch_size):
            batch = audio_files[i:i + batch_size]
            logging.info(f"Processing batch {i//batch_size + 1} of {(len(audio_files) + batch_size - 1)//batch_size}")
            batch_results = await process_batch(batch)
            # await asyncio.sleep(10)  # Prevent quota hit
            all_results.extend(batch_results)
            
            # 🛡 Optional cooldown between batches
            await asyncio.sleep(10)
        if not all_results:
            continue
            
        # Calculate processing time for this subdirectory
        total_time = time.time() - total_start
        
        # Add summary statistics
        summary = {
            "TotalFiles": total_files,
            "SuccessfulFiles": sum(1 for r in all_results if r["Success"]),
            "FailedFiles": sum(1 for r in all_results if not r["Success"]),
            # "TotalProcessingTime": round(total_time, 2),
            # "AverageTimePerFile": round(total_time / total_files, 2) if total_files else 0,
            "Results": all_results
        }
        
        # Save JSON using the subdirectory name
        subdir_name = os.path.basename(subdir)
        output_path = f"output/{subdir_name}.json"
        os.makedirs("output", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fout:
            json.dump(summary, fout, indent=2, ensure_ascii=False)
        logging.info(f"Results for {subdir_name} saved to {output_path}")
        
        if total_files:
            logging.info(f"Processed {total_files} files for {subdir_name} in {total_time:.1f}s (avg {total_time/total_files:.1f}s per file)")
    
    # Log final summary
    total_time = time.time() - total_start
    logging.info(f"Total processing completed in {total_time:.1f}s")
    logging.info(f"Processed files across {len(subdirs)} subdirectories")

if __name__ == "__main__":
    asyncio.run(main(directory="AudioFiles"))