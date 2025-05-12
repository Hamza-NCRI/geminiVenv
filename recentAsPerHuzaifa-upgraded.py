import os
import google.generativeai as genai
import time
import asyncio
import configparser
import json
import logging

# Configure logging
logging.basicConfig(filename='processing.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load API key from config
config = configparser.ConfigParser()
config.read('config.ini')
API_KEY = config['API_KEYS']['paidAPIKey']
genai.configure(api_key=API_KEY)

# Initialize the model
model = genai.GenerativeModel(
    'gemini-1.5-flash',
    generation_config={"response_mime_type": "application/json"}
)
# Define the JSON-based prompt
#  "CallTranscriptionSummary": string (Provide call transcription of max 150 words),
# prompt = """
# {
# Analyze the provided audio recording of a call, evaluating its content based on the structure and criteria outlined below. Ensure to extract key information and provide detailed assessments across the questions of the call listed below. The response should be formatted as a JSON object with the following structure:
# {
#       "Call Transcription Summary": string (Provide a concise summary of the call's key points, focusing on debt resolution strategies, payment plans, and customer interactions.),
#       "Customer Name": string (Extract the customer's name from the call conversation.),
#       "Agent Name": string (Extract the agent's name from the call conversation.),
#       "File Number": string (Retrieve the file number from the call, if applicable),
#       "Key Phrases": string (Main words in call like Car loan, Credit, Bank name etc),
#       "Positive Words": string(List all distinct positive words from the call conversation like appriciate, I can try etc),
#       "Negative Words": string(List all distinct negative words from the call conversation like stop calling, this is unfair),
#       "Neutral Words": string(List all distinct neutral words from the call conversation like stop maybe, I'm not sure okay),
#       "Overall Call Rating": integer (Rate the call between 1 and 5 based on the Sentiment Analysis 1 being lowest),
#       "IsPositiveVoiceToneAndPace": bool (Analyze voice and tone in the call like negative for raise voice etc positive for cooperation etc, true for positive)
#       "Call Outcome": int? (1 if a payment plan or settlement was agreed upon, 0 if no agreement was reached, or null if the call was inconclusive.),
#       "Call Outcome Message": string (Provide a brief explanation of the call outcome, including any specific agreements or next steps.),
#       "Did the agent address the customer by their first and last name, including middle initial if applicable?": int? (0 for Fail, 1 for Pass, else null),
#       "Did the agent identify themselves using their full name?": int? (0 for Fail, 1 for Pass, else null),
#       "Did the agent state the full company name (National Credit Recovery) OR NCRi abbreviation?": int? (0 for Fail, 1 for Pass, else null),
#       "Did the agent confirm the customer's date of birth or full address from the file?": int? (1 for Yes, 0 for No, else null),
#       "Did the agent advise the customer that all calls are recorded":int? (1 for Yes, 0 for No, else null),
#       "Did the agent specify the full name of the creditor in the disclosure (including original and current creditors if both are relevant)?": int? (1 for Yes, 0 for No, else null),
#       "Did the agent wait for the debtor's response to Payment In Full (PIF) before discussing other lending options?": int? (1 for Yes, 0 for No, else null),
#       "Did the agent ask for the reason behind the arrears and acknowledge the customer's current financial situation?": int? (1 for Yes, 0 for No, else null),      
#       "Did the agent negotiate a repayment or settlement plan that the customer accepted?": int? (1 for Yes, 0 for No,else null),
#       "Did the agent give clear and accurate advice or information to the customer?": int? (1 for Yes, 0 for No, else null),
#       "Did the agent secure a commitment from the customer, set appropriate expectations, and confirm conversation details?": int? (1 for Yes, 0 for No, else null),
#       "Did the agent set up a pre-authorized payment with the customer?": int? (1 for Yes, 0 for No, else null),
#       "Did the agent upsell or encourage a higher repayment amount?": int? (1 for Yes, 0 for No, else null),
#       "Did the agent discuss all available financing and lending options with the customer?": int? (1 for Yes, 0 for No, else null), 
#       "Did the agent use any available information, such as a credit report, to inform the discussion?": int? (1 for Yes, 0 for No, else null),
#       "Did the agent maintain a suitable tone and manner throughout the call?": int? (1 for Yes, 0 for No, else null),
#       "Did the agent actively listen and acknowledge the customer's concerns?": int? (1 for Yes, 0 for No, else null),
#       "Did the agent remain respectful and professional for the entire call?": int? (1 for Yes, 0 for No, else null),
#       "Did the agent positively represent both the client and the NCRi brand?": int? (1 for Yes, 0 for No, else null),
#       "Was the Overall Call was Outstanding?": int? (0 for Fail, 1 for Pass, else null),

# Please analyze the uploaded audio file, extract relevant data points, and complete the JSON structure based on your evaluation of the call.
# """
prompt = """
    Evaluate the following call transcript for quality assurance of a law firm's intake specialist. For each checkpoint, respond with a binary answer:
    "Yes" = Requirement clearly met
    "No" = Requirement not met
    "N/A" = Not applicable or not addressed
    Also provide a brief call summary (2–3 sentences) describing the caller's concern and how the intake specialist handled the situation.
    Return the result in JSON format, structured exactly as shown below. Ensure all fields are filled in with "Yes", "No", or "N/A" as appropriate.
        JSON Output Format:
        {
          "summary": "string",
          "section_1": {
            "greeted_warmly_and_slow": "Yes/No/N/A",
            "asked_personalization_question": "Yes/No/N/A"
          },
          "section_2": {
            "showed_empathy": "Yes/No/N/A",
            "controlled_call_flow": "Yes/No/N/A",
            "avoided_interruptions": "Yes/No/N/A",
            "used_caller_name": "Yes/No/N/A",
            "collected_phone_early": "Yes/No/N/A"
          },
          "section_3": {
            "asked_referral_source": "Yes/No/N/A",
            "advocated_for_firm": "Yes/No/N/A",
            "closed_with_thank_you": "Yes/No/N/A"
          }
        }
        Now evaluate the transcript accordingly.
"""

async def process_single_audio(audio_path, output_json, failed_files):
    start_time = time.time()
    try:
        # Upload the audio file
        audio_file = genai.upload_file(path=audio_path)
        if not audio_file:
            raise ValueError(f"Failed to upload file: {audio_path}")

        print(f"Processing: {audio_file.display_name}")
        logging.info(f"Processing file: {audio_file.display_name}")

        # Generate content
        response = model.generate_content([prompt, audio_file])
        if not response or not hasattr(response, 'text') or not response.text:
            raise ValueError(f"No valid response for {audio_file.display_name}")

        # Parse response as JSON
        try:
            response_data = json.loads(response.text)
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error for {audio_file.display_name}: {e}")
            failed_files.append(audio_path)
            return

        # Add extra metadata to the response
        response_data["FileName"] = audio_file.display_name
        response_data["FilePath"] = audio_path  # Store the full path
        response_data["Success"] = True
        response_data["Message"] = "Processed successfully."

        # Append response data to output JSON list
        output_json.append(response_data)

        # Delete the uploaded file
        genai.delete_file(audio_file.name)
        print(f"Deleted file {audio_file.display_name}")

    except Exception as e:
        logging.error(f"Error processing {audio_path}: {e}")
        failed_files.append(audio_path)

    end_time = time.time()
    print(f"Time taken for {audio_file.display_name}: {end_time - start_time:.2f}s")

def fetch_audio_files(base_directory):
    audio_files = []
    for root, _, files in os.walk(base_directory):
        for filename in files:
            if filename.endswith(('.mp3', '.wav')):
                audio_files.append(os.path.join(root, filename))
            else:
                logging.info(f"Excluded file: {filename}")
    return audio_files

async def process_audio_files(audio_paths, output_json, failed_files, batch_size=10):
    total_batches = (len(audio_paths) + batch_size - 1) // batch_size
    for i in range(0, len(audio_paths), batch_size):
        batch = audio_paths[i:i + batch_size]
        print(f"Starting batch {i // batch_size + 1}/{total_batches}")
        logging.info(f"Starting batch {i // batch_size + 1} with files: {batch}")
        tasks = [process_single_audio(audio, output_json, failed_files) for audio in batch]
        await asyncio.gather(*tasks)
    if failed_files:
        await retry_failed_files(failed_files, output_json)

async def retry_failed_files(failed_files, output_json):
    logging.info("Retrying failed files...")
    final_failed_files = []
    for audio_path in failed_files:
        try:
            await process_single_audio(audio_path, output_json, final_failed_files)
        except Exception as e:
            logging.error(f"Retry failed for {audio_path}: {e}")
            response_data = {
                "FileName": audio_path,
                "Success": False,
                "Message": f"Error processing file on retry: {e}"
            }
            output_json.append(response_data)

    # Update failed_files list for any files that failed the retry
    failed_files.clear()
    failed_files.extend(final_failed_files)

def organize_output_by_folders(output_json):
    folder_data = {}
    for item in output_json:
        if "FilePath" in item:
            # Get the second-level folder name (after 'test')
            path_parts = os.path.normpath(item["FilePath"]).split(os.sep)
            if len(path_parts) >= 2:  # Ensure there's at least 'test' and one subfolder
                # Assuming 'test' is the first part (index 0)
                if len(path_parts) >= 3:
                    folder_name = path_parts[1]  # The folder directly under 'test'
                    if folder_name not in folder_data:
                        folder_data[folder_name] = []
                    folder_data[folder_name].append(item)
    return folder_data

def save_output(output_json):
    # Organize data by folders
    folder_data = organize_output_by_folders(output_json)

    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Save separate JSON file for each folder
    for folder_name, data in folder_data.items():
        output_filename = os.path.join(output_dir, f"{folder_name}.json")
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logging.info(f"Output saved to {output_filename}")

if __name__ == "__main__":
    base_directory = "test"
    # base_directory = "PayPal Calls"
    audio_file_paths = fetch_audio_files(base_directory)
    print(f"Total Files: {len(audio_file_paths)}")

    if not audio_file_paths:
        print("No audio files found in the specified directory.")
    else:
        start_time = time.time()
        output_json = []
        failed_files = []

        asyncio.run(process_audio_files(audio_file_paths, output_json=output_json, failed_files=failed_files, batch_size=10))

        # Retry for files that failed in the first pass
        if failed_files:
            asyncio.run(retry_failed_files(failed_files, output_json))

        # Save final output JSON
        save_output(output_json)
        end_time = time.time()
        print(f"Total processing time: {end_time - start_time:.2f}s")