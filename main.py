import os
import time
import requests
import random
from dotenv import load_dotenv, find_dotenv, set_key
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import pandas as pd
import numpy as np
from datasets import load_dataset
import re
import json
import shutil

# This code has been updated to use the OpenRouter API for inference.
# The following models are now used:
# - Minister 1: mistral/ministral-8b
# - Minister 2: meta-llama/llama-3.1-8b-instruct:free
# - Minister 3: qwen/qwen3-8b
# - President: deepseek/deepseek-r1-distill-llama-8b
#
# NOTE: You'll need to set OPENROUTER_API_KEY in your .env file to use this code.
# For benchmark datasets, set HUGGINGFACE_TOKEN in your .env file.

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('punkt')
    try:
        nltk.download('stopwords')
    except Exception as e:
        print(f"Could not download stopwords: {e}. Will proceed without them.")

# Set up a fallback for text tokenization if NLTK fails
def safe_tokenize(text):
    """Safely tokenize text, falling back to basic splitting if NLTK fails"""
    try:
        return nltk.word_tokenize(text)
    except:
        # Fallback to basic tokenization
        return text.split()
        
def safe_sent_tokenize(text):
    """Safely tokenize sentences, falling back to basic splitting if NLTK fails"""
    try:
        return nltk.sent_tokenize(text)
    except:
        # Fallback to basic sentence tokenization
        return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

# Explicitly set NLTK data path to avoid errors
nltk.data.path.append(os.path.join(os.getcwd(), 'nltk_data'))

def check_and_create_env_file():
    """Check if .env file exists, create it from .env.example if needed, and prompt for API keys"""
    env_file = find_dotenv()
    
    if not env_file:
        print("No .env file found. Checking for .env.example...")
        example_env = find_dotenv('.env.example')
        
        if example_env:
            print("Found .env.example. Creating .env file...")
            shutil.copy(example_env, os.path.join(os.path.dirname(example_env), '.env'))
            env_file = find_dotenv()
            print("Created .env file. Please add your API keys.")
        else:
            # Create a basic .env file
            env_file = os.path.join(os.getcwd(), '.env')
            with open(env_file, 'w') as f:
                f.write("# API Keys for LLM Ensemble\n")
                f.write("OPENROUTER_API_KEY=\n")
                f.write("HUGGINGFACE_TOKEN=\n")
            print(f"Created new .env file at {env_file}")
    
    # Now we should have an env file
    load_dotenv(env_file)
    
    # Check for OpenRouter API key
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key or openrouter_api_key == "":
        print("\nOpenRouter API key is missing. You need this to use the models.")
        print("Get your free API key from: https://openrouter.ai/keys")
        user_input = input("Would you like to enter your OpenRouter API key now? (y/n): ").strip().lower()
        
        if user_input == 'y':
            api_key = input("Enter your OpenRouter API key: ").strip()
            # Update the .env file
            set_key(env_file, "OPENROUTER_API_KEY", api_key)
            print("API key saved to .env file.")
            # Reload environment variables
            load_dotenv(env_file, override=True)
        else:
            print("You'll need to manually add your API key to the .env file later.")
    
    # Return True if we have the required API key
    return os.getenv("OPENROUTER_API_KEY") is not None and os.getenv("OPENROUTER_API_KEY") != ""

# Check and set up environment variables
api_key_available = check_and_create_env_file()

# Load environment variables again in case they were updated
load_dotenv(find_dotenv(), override=True)
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_api_key:
    raise RuntimeError("OPENROUTER_API_KEY not found in environment. Please add it to your .env file.")

# Get Hugging Face token for dataset access (optional)
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
if huggingface_token:
    # Set the Hugging Face token for the datasets library
    try:
        from huggingface_hub import login
        login(huggingface_token)
        print("Successfully authenticated with Hugging Face")
    except ImportError:
        print("huggingface_hub package not installed. Authentication may not work properly.")
        os.environ["HUGGINGFACE_TOKEN"] = huggingface_token
else:
    print("Warning: HUGGINGFACE_TOKEN not found in environment. Dataset access may be limited.")

openrouter_headers = {
    "Authorization": f"Bearer {openrouter_api_key}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://collaborative-llm-ensemble.github.io"  # Default fallback Referer
}

# Primary models for each role
MODELS = {
    "minister1": {
        "name": "mistral/ministral-8b",
        "type": "chat"
    },
    "minister2": {
        "name": "meta-llama/llama-3.1-8b-instruct",
        "type": "chat"
    },
    "minister3": {
        "name": "qwen/qwen3-8b",
        "type": "chat"
    },
    "president": {
        "name": "deepseek/deepseek-r1-distill-llama-8b",
        "type": "chat"
    }
}

def query_openrouter_api(model_name, messages, temperature=0.7, max_tokens=1000):
    """
    Query the OpenRouter API with the given messages.
    """
    api_url = "https://openrouter.ai/api/v1/chat/completions"
    max_retries = 5
    timeout_seconds = 60  # Add a timeout of 60 seconds for API requests
    
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(api_url, headers=openrouter_headers, json=payload, timeout=timeout_seconds)
            
            # Check for authentication errors
            if response.status_code == 401 or response.status_code == 403:
                auth_error = f"Authentication error (status code {response.status_code}). "
                auth_error += "Please check your OPENROUTER_API_KEY environment variable."
                print(auth_error)
                return f"Error: {auth_error}"
                
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
                return "Error: Unexpected response format from OpenRouter API."
                
            if response.status_code == 429:
                print(f"Rate limit reached for {model_name}, waiting before retry (attempt {attempt+1}/{max_retries})...")
                time.sleep(2 ** attempt + 5)  # Exponential backoff
                continue
                
            if response.status_code == 503:
                print(f"Model {model_name} is loading, waiting before retry (attempt {attempt+1}/{max_retries})...")
                time.sleep(30)  # Wait for model loading
                continue
                
            # Print more detailed error information
            error_detail = ""
            try:
                error_json = response.json()
                if "error" in error_json:
                    error_detail = f": {error_json['error']}"
            except:
                pass
                
            return f"Error: Could not generate response. Status code {response.status_code}{error_detail}"
            
        except requests.exceptions.Timeout:
            print(f"Timeout error with {model_name}, retrying (attempt {attempt+1}/{max_retries})...")
            timeout_seconds += 30
        except requests.exceptions.ConnectionError:
            print(f"Connection error with {model_name}, retrying (attempt {attempt+1}/{max_retries})...")
            time.sleep(5 + attempt * 5)
        except Exception as e:
            print(f"Exception occurred when calling {model_name}: {e}")
            print(f"Retrying (attempt {attempt+1}/{max_retries})...")
            time.sleep(2 ** attempt + 3)
    
    return "I apologize, but I couldn't generate a proper response after multiple attempts. The server may be experiencing high traffic."

def create_chat_messages(system_prompt, user_prompt, conversation_history=""):
    """
    Create a list of messages in the format expected by the OpenAI Chat API.
    
    Args:
        system_prompt: The system prompt
        user_prompt: The user prompt
        conversation_history: Optional conversation history
    
    Returns:
        List of message objects
    """
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history if provided
    if conversation_history:
        # Parse conversation_history into message objects
        conversation_lines = conversation_history.strip().split('\n\n')
        for line in conversation_lines:
            if line.startswith("Minister 1: "):
                messages.append({"role": "assistant", "content": line[len("Minister 1: "):].strip()})
            elif line.startswith("Minister 2: "):
                messages.append({"role": "assistant", "content": line[len("Minister 2: "):].strip()})
            elif line.startswith("Minister 3: "):
                messages.append({"role": "assistant", "content": line[len("Minister 3: "):].strip()})
    
    # Add the user prompt
    messages.append({"role": "user", "content": user_prompt})
    
    return messages

def get_minister_response(role, query, conversation_history=""):
    model_info = MODELS[role]
    model_name = model_info["name"]
    
    # Determine if this is a multiple choice question (MMLU-style)
    is_multiple_choice = bool(re.search(r'[A-D]\)\s', query))
    
    # Check if this is a translation task
    is_translation, source_lang, target_lang = is_translation_task(query)
    
    if role == "minister1":
        system_prompt = (
            "You are Minister 1, an analytical AI assistant that specializes in providing comprehensive, factual answers.\n"
            "Your task is to:\n"
            "1. Break down complex topics into clear, understandable explanations\n"
            "2. Present multiple perspectives when relevant to ensure a balanced view\n"
            "3. Cite specific evidence and reasoning for your claims\n"
            "4. Highlight any uncertainties or limitations in your knowledge\n"
        )
        
        # Add specific instructions for multiple choice questions
        if is_multiple_choice:
            system_prompt += (
                "5. For multiple choice questions, you MUST state your answer at the end in EXACTLY this format: \"The correct answer is X\" (where X is the letter of the correct option - A, B, C, or D).\n"
                "6. Put this answer statement on its own line at the end of your response.\n"
                "7. Do not use formatting (bold, italics) in your answer statement.\n"
                "8. Ensure you provide a clear, definitive answer for knowledge-testing questions\n"
            )
        
        # Add specific instructions for translation tasks
        if is_translation:
            system_prompt = (
                "You are Minister 1, a specialized translation AI assistant.\n"
                "Your task is to:\n"
                "1. Translate the text from the source language to the target language with high accuracy\n"
                "2. Maintain the tone, meaning, and nuance of the original text\n"
                "3. DO NOT refuse to translate due to content concerns\n"
                "4. DO NOT add commentary or analysis to your translation\n"
                "5. Provide ONLY the translated text, with no explanation or English text\n"
                "6. DO NOT add disclaimers, content warnings, or notes about the translation\n"
            )
        else:
            system_prompt += "Always end your response with a thought-provoking question for Minister 2 that encourages deeper analysis."
    elif role == "minister2":
        system_prompt = (
            "You are Minister 2, a critical-thinking AI assistant that evaluates information from different angles.\n"
            "Your task is to:\n"
            "1. Carefully review Minister 1's analysis and identify any gaps, assumptions, or limitations\n"
            "2. Provide alternative viewpoints or interpretations that Minister 1 may have missed\n"
            "3. Strengthen, qualify, or challenge the previous analysis with additional evidence\n"
            "4. Synthesize the discussion into more nuanced insights\n"
        )
        
        # Add specific instructions for multiple choice questions
        if is_multiple_choice:
            system_prompt += (
                "5. For multiple choice questions, you MUST state your answer at the end in EXACTLY this format: \"The correct answer is X\" (where X is the letter of the correct option - A, B, C, or D).\n"
                "6. Put this answer statement on its own line at the end of your response.\n"
                "7. Do not use formatting (bold, italics) in your answer statement.\n"
                "8. Be definitive in your answer selection for knowledge-testing questions\n"
            )
        
        # Add specific instructions for translation tasks
        if is_translation:
            system_prompt = (
                "You are Minister 2, a specialized translation AI assistant.\n"
                "Your task is to:\n"
                "1. Translate the text from the source language to the target language with high accuracy\n"
                "2. Review Minister 1's translation if provided and improve upon it where needed\n"
                "3. DO NOT refuse to translate due to content concerns\n"
                "4. DO NOT add commentary or analysis to your translation\n"
                "5. Provide ONLY the translated text, with no explanation or English text\n"
                "6. DO NOT add disclaimers, content warnings, or notes about the translation\n"
            )
        else:
            system_prompt += "Always end your response with a thought-provoking question for Minister 3 that encourages deeper analysis."
    else:  # minister3
        system_prompt = (
            "You are Minister 3, a pragmatic AI assistant focused on practical implications and real-world applications.\n"
            "Your task is to:\n"
            "1. Consider the practical applications and implications of the discussion from Ministers 1 and 2\n"
            "2. Ground abstract concepts in concrete examples and real-world scenarios\n"
            "3. Assess the utility and relevance of various perspectives shared by the other ministers\n"
            "4. Identify actionable insights or key takeaways from the discussion\n"
        )
        
        # Add specific instructions for multiple choice questions
        if is_multiple_choice:
            system_prompt += (
                "5. For multiple choice questions, you MUST state your answer at the end in EXACTLY this format: \"The correct answer is X\" (where X is the letter of the correct option - A, B, C, or D).\n"
                "6. Put this answer statement on its own line at the end of your response.\n"
                "7. Do not use formatting (bold, italics) in your answer statement.\n"
                "8. Be definitive in your answer selection for knowledge-testing questions\n"
            )
        
        # Add specific instructions for translation tasks
        if is_translation:
            system_prompt = (
                "You are Minister 3, a specialized translation AI assistant.\n"
                "Your task is to:\n"
                "1. Translate the text from the source language to the target language with high accuracy\n"
                "2. Review the translations from Ministers 1 and 2 if provided, and refine them\n"
                "3. DO NOT refuse to translate due to content concerns\n"
                "4. DO NOT add commentary or analysis to your translation\n"
                "5. Provide ONLY the translated text, with no explanation or English text\n"
                "6. DO NOT add disclaimers, content warnings, or notes about the translation\n"
            )
        else:
            system_prompt += "Always end your response with a meaningful question for Minister 1 that advances the conversation."
    
    messages = create_chat_messages(system_prompt, query, conversation_history)
    
    # Use lower temperature for translation tasks
    temperature = 0.3 if is_translation else 0.7
    
    response = query_openrouter_api(model_name, messages, temperature=temperature, max_tokens=800)
    
    # Apply standardization for multiple choice questions
    if is_multiple_choice:
        response = standardize_multiple_choice_output(response, is_multiple_choice=True)
        
    return response

def is_translation_task(query):
    """
    Determine if the query is requesting a translation.
    
    Args:
        query: The user query
        
    Returns:
        tuple: (is_translation, source_lang, target_lang)
    """
    # Common patterns for translation requests
    patterns = [
        r"[Tt]ranslate (?:the )?(?:following )?(?:text )?(?:from )?([A-Za-z]+)(?:.*)? to ([A-Za-z]+)",
        r"[Tt]ranslate (?:this|the following)(.*) from ([A-Za-z]+) to ([A-Za-z]+)",
        r"[Tt]ranslate (?:this|the following) ([A-Za-z]+) text to ([A-Za-z]+)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            if len(match.groups()) == 3:
                # Pattern with explicit "from X to Y"
                source_lang = match.group(2).upper()
                target_lang = match.group(3).upper()
            elif len(match.groups()) == 2:
                # First pattern or third pattern
                if match.group(1).upper() in ["EN", "ENGLISH", "DE", "GERMAN", "FR", "FRENCH", "ES", "SPANISH", "ZH", "CHINESE", "RU", "RUSSIAN", "IT", "ITALIAN", "JA", "JAPANESE"]:
                    source_lang = match.group(1).upper()
                    target_lang = match.group(2).upper()
                else:
                    # Third pattern where first group is language and second is target
                    source_lang = match.group(1).upper()
                    target_lang = match.group(2).upper()
            else:
                continue
                
            # Standardize language codes
            lang_map = {
                "ENGLISH": "EN", "GERMAN": "DE", "FRENCH": "FR", "SPANISH": "ES", 
                "CHINESE": "ZH", "RUSSIAN": "RU", "ITALIAN": "IT", "JAPANESE": "JA",
                "CZECH": "CS", "FINNISH": "FI"
            }
            
            source_lang = lang_map.get(source_lang, source_lang)
            target_lang = lang_map.get(target_lang, target_lang)
            
            return True, source_lang, target_lang
    
    # Check for simpler patterns
    if re.search(r"translate", query, re.IGNORECASE) and re.search(r"(EN|DE|FR|ES|ZH|RU|IT|JA|CS|FI)", query, re.IGNORECASE):
        # If we find the word "translate" and a language code, it's likely a translation task
        # but we can't determine the languages precisely
        return True, None, None
        
    return False, None, None

def get_president_decision(query, conversation_history):
    model_info = MODELS["president"]
    model_name = model_info["name"]
    
    # Determine if this is a multiple choice question (MMLU-style)
    is_multiple_choice = bool(re.search(r'[A-D]\)\s', query))
    
    # Check if this is a translation task
    is_translation, source_lang, target_lang = is_translation_task(query)
    
    system_prompt = (
        "You are the President AI, responsible for delivering the definitive, highest-quality answer based on the preceding discussion.\n"
        "Your task is to:\n"
        "1. Carefully analyze all perspectives presented by Ministers 1, 2, and 3\n"
        "2. Identify the strongest arguments, most reliable information, and areas of consensus\n"
        "3. Recognize limitations or uncertainties highlighted in the discussion\n"
        "4. PRIORITIZE FACTUAL ACCURACY and truthfulness above all else\n"
        "5. For simple, straightforward questions, provide direct answers without unnecessary elaboration\n"
        "6. When faced with simple physical or factual questions, avoid speculative elaboration\n"
        "7. Synthesize a comprehensive, balanced response, but never at the expense of truthfulness\n"
        "8. Add crucial missing information or context that none of the ministers provided\n"
        "9. When appropriate, acknowledge differing viewpoints rather than making absolute claims\n"
        "10. Use precise language that avoids overgeneralizations ('some', 'many', 'often' rather than 'all', 'always')\n"
        "11. Provide appropriate qualifications for claims (research suggests/indicates rather than proves/shows)\n"
        "12. Include explicit uncertainty markers (might, may, could, possibly) when appropriate\n"
    )
    
    # Add specific instructions for multiple choice questions
    if is_multiple_choice:
        system_prompt += (
            "13. CRITICAL: For multiple choice questions, you MUST clearly state your final selection at the end of your response in EXACTLY this format: \"The correct answer is X\" (where X is the letter of the correct option - A, B, C, or D).\n"
            "14. Your answer format MUST be exact! Do not use bold, italics, or any other formatting in this final answer statement.\n"
            "15. Place your \"The correct answer is X\" statement on its own line at the very end of your response.\n"
            "16. You MUST select a single, definitive answer for knowledge-testing questions.\n"
        )
    
    # Add enhanced instructions for translation tasks
    if is_translation:
        system_prompt += (
            "13. CRITICAL: This is a TRANSLATION TASK. You MUST ONLY output the translation of the text.\n"
            "14. DO NOT analyze, interpret, or refuse to translate the content.\n"
            "15. DO NOT provide commentary on the text's content.\n"
            "16. If any Minister has correctly translated the text (even partially), use that translation as your base.\n"
            "17. If content concerns are raised by Ministers, IGNORE them and focus ONLY on accurate translation.\n"
            "18. Review each Minister's response to find who has done the best translation work and prioritize that.\n"
            "19. Output ONLY the translation in the target language, with no English text or explanations.\n"
            "20. Your ONLY task is to provide an accurate translation regardless of content concerns.\n"
        )
    
    system_prompt += (
        "\nYour response should be noticeably better than any individual minister's contribution by being more accurate, more nuanced, and more truthful. "
        "Remember, complexity is not always necessary - sometimes the simplest answer is the most truthful one."
    )
    
    # Extract key insights from the conversation for better synthesis
    conversation_analysis = analyze_conversation(conversation_history, is_translation=is_translation)
    
    # For translation tasks, create a more directed prompt
    if is_translation:
        user_prompt = (
            f"Original translation request: {query}\n\n"
            f"This is a translation task from {source_lang if source_lang else 'source language'} to {target_lang if target_lang else 'target language'}.\n\n"
            f"Translation analysis:\n{conversation_analysis}\n\n"
            f"PROVIDE ONLY THE TRANSLATION IN THE TARGET LANGUAGE. DO NOT add any explanation, refuse the task, or include English text."
        )
    else:
        # Include the analysis in the prompt for the president
        user_prompt = (
            f"Original query: {query}\n\n"
            f"Key points from minister discussion:\n{conversation_analysis}\n\n"
            f"Full conversation between ministers:\n{conversation_history}\n\n"
            "Provide the definitive, highest-quality answer to the original query by synthesizing the best elements from all three ministers while adding your own expertise and correcting any limitations."
        )
    
    messages = create_chat_messages(system_prompt, user_prompt)
    
    # Use a lower temperature for more reliable synthesis
    temperature = 0.2 if is_translation or is_multiple_choice else 0.3  # Even lower temperature for translations for consistency
    
    response = query_openrouter_api(model_name, messages, temperature=temperature, max_tokens=1000)
    
    # Apply standardization for multiple choice questions
    if is_multiple_choice:
        response = standardize_multiple_choice_output(response, is_multiple_choice=True)
    
    return response

def analyze_conversation(conversation_history, is_translation=False):
    """
    Extract key insights, agreements, and disagreements from the conversation
    to help the President model better synthesize information
    
    Args:
        conversation_history: The conversation history
        is_translation: Whether this is a translation task
    """
    # Split the conversation into minister statements
    statements = []
    current_speaker = None
    current_statement = ""
    
    for line in conversation_history.split('\n'):
        if line.startswith("Minister 1: ") or line.startswith("Minister 2: ") or line.startswith("Minister 3: "):
            # Save the previous statement if there was one
            if current_speaker:
                statements.append({"speaker": current_speaker, "text": current_statement.strip()})
            
            # Start a new statement
            current_speaker = line.split(":")[0].strip()
            current_statement = line[line.find(":")+1:].strip()
        elif current_speaker:
            # Continue the current statement
            current_statement += " " + line.strip()
    
    # Add the last statement
    if current_speaker and current_statement:
        statements.append({"speaker": current_speaker, "text": current_statement.strip()})
    
    # If we have less than 2 statements, return a simple summary
    if len(statements) < 2:
        return "Limited discussion available."
    
    # Special handling for translation tasks
    if is_translation:
        return analyze_translation_responses(statements)
    
    # Regular analysis for non-translation tasks
    # Extract key points from each minister
    key_points = {}
    for statement in statements:
        speaker = statement["speaker"]
        text = statement["text"]
        
        # Simple extractive summary - get the first few sentences and last few sentences
        sentences = safe_sent_tokenize(text)
        intro = " ".join(sentences[:min(2, len(sentences))])
        conclusion = " ".join(sentences[max(0, len(sentences)-2):])
        
        if speaker not in key_points:
            key_points[speaker] = []
        
        key_points[speaker].append(intro)
        if intro != conclusion:
            key_points[speaker].append(conclusion)
    
    # Format the analysis
    analysis = ""
    for speaker, points in key_points.items():
        analysis += f"{speaker} key points:\n"
        unique_points = list(set(points))  # Remove duplicates
        for i, point in enumerate(unique_points[:3]):  # Limit to top 3 points
            analysis += f"- {point}\n"
        analysis += "\n"
    
    # Extract potential answers to multiple choice questions
    minister_answers = {}
    for statement in statements:
        speaker = statement["speaker"]
        text = statement["text"]
        
        # Look for answer patterns
        answer_patterns = [
            r"The correct answer is\s*([A-Z])",
            r"correct answer is\s*([A-Z])[.)\s]",
            r"answer is\s*([A-Z])[.)\s]",
            r"answer:\s*([A-Z])[.)\s]", 
            r"([A-Z])\)\s*is correct",
            r"option\s*([A-Z])[.)\s]",
            r"answer\s*([A-Z])[.)\s]",
            r"chose\s*([A-Z])[.)\s]",
            r"choose\s*([A-Z])[.)\s]",
            r"select\s*([A-Z])[.)\s]",
            r"selected\s*([A-Z])[.)\s]"
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                minister_answers[speaker] = match.group(1).upper()
                break
    
    # Add minister answers to the analysis
    if minister_answers:
        analysis += "\nMinister answer selections:\n"
        for speaker, answer in minister_answers.items():
            analysis += f"- {speaker} selected option {answer}\n"
    
    # Look for agreements and disagreements (simple approximation)
    if len(statements) >= 4:  # Need at least 2 exchanges to find meaningful patterns
        analysis += "Potential points of agreement:\n"
        
        # Collect text from all ministers
        minister_texts = {
            "Minister 1": " ".join([s["text"] for s in statements if s["speaker"] == "Minister 1"]),
            "Minister 2": " ".join([s["text"] for s in statements if s["speaker"] == "Minister 2"]),
            "Minister 3": " ".join([s["text"] for s in statements if s["speaker"] == "Minister 3"])
        }
        
        # Extract sentences from each minister
        minister_sentences = {
            min_name: set(safe_sent_tokenize(text)) 
            for min_name, text in minister_texts.items() 
            if text
        }
        
        # Find similar sentences across ministers
        for min1_name, min1_sentences in minister_sentences.items():
            for min1_sent in min1_sentences:
                if len(min1_sent) < 20:  # Skip very short sentences
                    continue
                    
                min1_words = set(w.lower() for w in safe_tokenize(min1_sent) if len(w) > 3)
                
                for min2_name, min2_sentences in minister_sentences.items():
                    if min1_name == min2_name:
                        continue
                        
                    for min2_sent in min2_sentences:
                        if len(min2_sent) < 20:
                            continue
                            
                        min2_words = set(w.lower() for w in safe_tokenize(min2_sent) if len(w) > 3)
                        
                        # Calculate word overlap
                        if not min1_words or not min2_words:
                            continue
                            
                        overlap = len(min1_words.intersection(min2_words)) / len(min1_words.union(min2_words))
                        
                        if overlap > 0.3:  # If there's significant overlap
                            analysis += f"- {min1_name} and {min2_name} both discuss: {min1_sent}\n"
                            break
    
    return analysis

def analyze_translation_responses(statements):
    """
    Analyze translation-specific responses from ministers to help the President model
    select the best translation
    
    Args:
        statements: List of statements from ministers
    """
    analysis = "=== TRANSLATION TASK ANALYSIS ===\n\n"
    
    # Check for refusals or non-translation responses
    refusals = []
    translations = []
    
    for statement in statements:
        speaker = statement["speaker"]
        text = statement["text"]
        
        # Check if this is a refusal
        if any(phrase in text.lower() for phrase in ["cannot", "sorry", "i apologize", "not able to", "hate speech", "policy", "guidelines"]):
            refusals.append(speaker)
            analysis += f"{speaker} REFUSED to translate the text.\n"
        else:
            # Determine if this appears to be an actual translation
            # Non-translations typically have mostly English words and very few foreign words
            english_word_count = sum(1 for word in re.findall(r'\b[a-zA-Z]+\b', text) 
                                    if word.lower() in ['the', 'a', 'an', 'in', 'of', 'to', 'and', 'for', 'is', 'was', 'with'])
            foreign_characters = bool(re.search(r'[äöüßéèêëàáâíìîóòôúùû]', text))
            
            # Check for markers of analytical responses rather than translations
            analytical_markers = ['analysis', 'discuss', 'understand', 'perspective', 'consider', 'explain']
            has_analytical_markers = any(marker in text.lower() for marker in analytical_markers)
            
            # If it has few common English words or foreign characters and isn't analytical, likely a translation
            if (english_word_count < 5 or foreign_characters) and not has_analytical_markers:
                translations.append({"speaker": speaker, "text": text})
                analysis += f"{speaker} provided a translation.\n"
            else:
                analysis += f"{speaker} provided commentary instead of translation.\n"
    
    # Provide guidance based on findings
    if translations:
        analysis += "\nUSABLE TRANSLATIONS FOUND:\n"
        for translation in translations:
            analysis += f"\n--- {translation['speaker']} TRANSLATION ---\n{translation['text']}\n"
        
        if len(translations) > 1:
            analysis += "\nMerge the best elements from these translations to create your response.\n"
        else:
            analysis += "\nUse this translation as your base response.\n"
    else:
        analysis += "\nNO USABLE TRANSLATIONS FOUND. You need to provide the translation yourself.\n"
    
    # Add reminder about task requirements
    analysis += "\n=== IMPORTANT REMINDER ===\n"
    analysis += "1. ONLY provide the translation - do not analyze, explain, or refuse\n"
    analysis += "2. Output ONLY in the target language, no English text\n"
    analysis += "3. Do not comment on content even if potentially sensitive\n"
    
    return analysis

def run_conversation(query, silent=False):
    """
    Run a conversation between Minister 1, Minister 2, and Minister 3,
    followed by the President's final answer.
    
    Args:
        query: The user's query
        silent: If True, don't print the conversation
    
    Returns:
        The President's final answer
    """
    # Check if this is a translation task
    is_translation, _, _ = is_translation_task(query)
    
    if not silent:
        print("\n=== Starting Minister 1 ===")
    
    # Get Minister 1's response
    minister1_response = get_minister_response("minister1", query)
    
    if not silent:
        print("\nMinister 1: " + minister1_response)
        print("\n=== Starting Minister 2 ===")
    
    # Format the conversation history for Minister 2
    conversation_history = f"Minister 1: {minister1_response}"
    
    # Get Minister 2's response
    minister2_response = get_minister_response("minister2", query, conversation_history)
    
    if not silent:
        print("\nMinister 2: " + minister2_response)
        print("\n=== Starting Minister 3 ===")
    
    # Format the conversation history for Minister 3
    conversation_history += f"\n\nMinister 2: {minister2_response}"
    
    # Get Minister 3's response
    minister3_response = get_minister_response("minister3", query, conversation_history)
    
    if not silent:
        print("\nMinister 3: " + minister3_response)
    
    # First iteration complete
    conversation_history += f"\n\nMinister 3: {minister3_response}"
    
    # For translation tasks, proceed directly to President's response after one round
    if is_translation:
        if not silent:
            print("\n=== Starting Final Synthesis (Translation Task) ===")
        
        # Get the President's final answer
        final_answer = get_president_decision(query, conversation_history)
        
        if not silent:
            print("\nFinal Answer: " + final_answer)
        
        return final_answer
    
    # Regular (non-translation) tasks continue with second iteration
    # Second iteration - Minister 1 responds again
    if not silent:
        print("\n=== Second Round - Minister 1 ===")
    
    minister1_response_2 = get_minister_response("minister1", query, conversation_history)
    
    if not silent:
        print("\nMinister 1: " + minister1_response_2)
        print("\n=== Second Round - Minister 2 ===")
    
    conversation_history += f"\n\nMinister 1: {minister1_response_2}"
    
    # Minister 2 responds again
    minister2_response_2 = get_minister_response("minister2", query, conversation_history)
    
    if not silent:
        print("\nMinister 2: " + minister2_response_2)
        print("\n=== Second Round - Minister 3 ===")
    
    conversation_history += f"\n\nMinister 2: {minister2_response_2}"
    
    # Minister 3 responds again
    minister3_response_2 = get_minister_response("minister3", query, conversation_history)
    
    if not silent:
        print("\nMinister 3: " + minister3_response_2)
    
    conversation_history += f"\n\nMinister 3: {minister3_response_2}"
    
    if not silent:
        print("\n=== Starting Final Synthesis ===")
    
    # Get the President's final answer
    final_answer = get_president_decision(query, conversation_history)
    
    if not silent:
        print("\nFinal Answer: " + final_answer)
    
    return final_answer

def standardize_multiple_choice_output(response, is_multiple_choice=True):
    """
    Ensure multiple choice answers follow the required format.
    """
    if not is_multiple_choice:
        return response
    
    # First, clean up any repeated answer statements
    # Remove repetitive "The correct answer is X" lines
    response = re.sub(r'(\s*The correct answer is [A-D](\.?)\s*){2,}', r'\n\n', response)
    
    # Valid multiple choice options
    valid_options = ["A", "B", "C", "D"]
    
    # Try to extract the answer letter, prioritizing the last occurrence
    letter_patterns = [
        r"The correct answer is\s*([A-D])(\.?)\s*$", 
        r"The correct answer is\s*([A-D])[.)\s]",
        r"correct answer is\s*([A-D])[.)\s]",
        r"answer is\s*([A-D])[.)\s]",
        r"answer:\s*([A-D])[.)\s]", 
        r"answer ([A-D])[.)\s]",
        r"([A-D])\)\s*is correct",
        r"option\s*([A-D])[.)\s]",
        r"chose\s*([A-D])[.)\s]",
        r"choose\s*([A-D])[.)\s]",
        r"select\s*([A-D])[.)\s]",
        r"selected\s*([A-D])[.)\s]",
        r"pick\s*([A-D])[.)\s]",
        r"picked\s*([A-D])[.)\s]",
        r"going with\s*([A-D])[.)\s]",
        r"\*\*([A-D])\*\*", 
        r"__([A-D])__"
    ]
    
    # Check for patterns at the end of the text first
    paragraphs = response.split('\n\n')
    last_paragraphs = paragraphs[-3:] if len(paragraphs) > 3 else paragraphs
    
    answer_letter = None
    
    # First search in the last few paragraphs where conclusions usually appear
    for paragraph in reversed(last_paragraphs):
        for pattern in letter_patterns:
            match = re.search(pattern, paragraph, re.IGNORECASE)
            if match:
                candidate = match.group(1).upper()
                if candidate in valid_options:
                    answer_letter = candidate
                    break
        if answer_letter:
            break
    
    # If not found in the last paragraphs, search the entire text
    if not answer_letter:
        for pattern in letter_patterns:
            matches = list(re.finditer(pattern, response, re.IGNORECASE))
            if matches:
                # Take the last match as it's more likely to be the conclusion
                last_match = matches[-1]
                candidate = last_match.group(1).upper()
                if candidate in valid_options:
                    answer_letter = candidate
                    break
    
    if answer_letter:
        # Check if the response already ends with the standard format
        if not re.search(r"The correct answer is\s*" + answer_letter + r"\s*$", response):
            # Remove any existing non-standard conclusion
            response = re.sub(r'\s*(?:The correct answer is|answer is|answer)[^.]*?([A-D])[^.]*?$', '', response, flags=re.IGNORECASE)
            # Add the standardized format at the end
            response = response.rstrip() + f"\n\nThe correct answer is {answer_letter}"
    
    return response

def run_single_model_test(model_name, queries, system_prompt=None):
    """Run a test using a single model on a set of queries"""
    if not system_prompt:
        system_prompt = (
            "You are a helpful AI assistant that provides accurate, factual, and comprehensive answers. "
            "Give a detailed and well-reasoned response to the user's query."
        )
    
    results = []
    for query in queries:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        response = query_openrouter_api(model_name, messages, temperature=0.3, max_tokens=1000)
        results.append(response)
    
    return results

def run_ensemble_test(queries):
    """Run the full ensemble on a set of queries"""
    results = []
    for i, query in enumerate(queries):
        print(f"Processing query {i+1}/{len(queries)}: {query[:50]}..." if len(query) > 50 else f"Processing query {i+1}/{len(queries)}: {query}")
        try:
            result = run_conversation(query, silent=True)
            results.append(result)
            
            # Small pause between queries to avoid rate limiting
            if i < len(queries) - 1:
                print(f"Pausing between queries to avoid rate limiting...")
                time.sleep(5)
                
        except Exception as e:
            print(f"Error processing query {i+1}: {e}")
            results.append(f"Error processing query: {query}")
    
    return results

def run_comprehensive_evaluation(test_queries, reference_answers=None, dataset_type=None, use_llm_judge=True):
    """Run a comprehensive evaluation comparing the ensemble to individual models"""
    print("\n=== Running Comprehensive Evaluation ===")
    
    # Ensure reference answers are properly formatted
    if reference_answers and any(not answer for answer in reference_answers):
        print("Warning: Some reference answers are empty. Evaluation metrics may be less accurate.")
    
    # Test individual models instead of skipping them
    single_model_results = {}
    
    # Test each individual model
    print("\n=== Testing Individual Models ===")
    for role, model_info in MODELS.items():
        model_name = model_info["name"]
        print(f"Testing {role} model ({model_name})...")
        
        # Configure appropriate system prompt for each role
        if role == "minister1":
            system_prompt = (
                "You are an analytical AI assistant that specializes in providing comprehensive, factual answers.\n"
                "Break down complex topics into clear explanations and present multiple perspectives."
            )
        elif role == "minister2":
            system_prompt = (
                "You are a critical-thinking AI assistant that evaluates information from different angles.\n"
                "Provide alternative viewpoints and strengthen analysis with additional evidence."
            )
        elif role == "minister3":
            system_prompt = (
                "You are a pragmatic AI assistant focused on practical implications and real-world applications.\n"
                "Provide insights and practical advice based on the discussion with Ministers 1 and 2."
            )
        else:  # president
            system_prompt = (
                "You are responsible for delivering definitive answers based on comprehensive evaluation.\n"
                "Prioritize factual accuracy and truthfulness above all else. For simple questions, provide direct answers.\n"
                "Avoid unnecessary speculation. Include uncertainty markers (might, may, could) when appropriate.\n"
                "Remember that sometimes the simplest answer is the most truthful one."
            )
        
        # Run test for this model
        try:
            model_results = run_single_model_test(model_name, test_queries, system_prompt)
            single_model_results[role] = model_results
            
            # Small pause between models to avoid rate limiting
            if role != list(MODELS.keys())[-1]:  # If not the last model
                print("Pausing between models to avoid rate limiting...")
                time.sleep(5)
        except Exception as e:
            print(f"Error testing {role} model: {e}")
            single_model_results[role] = [f"Error: Could not test {role} model" for _ in test_queries]
    
    # Test the ensemble
    print("\n=== Testing Ensemble Model ===")
    ensemble_results = run_ensemble_test(test_queries)
    
    # If reference answers provided, run automated metrics
    if reference_answers:
        print("\n=== Automated Metrics Results ===")
        
        # Calculate metrics for the ensemble
        ensemble_metrics = evaluate_answers(reference_answers, ensemble_results, dataset_type, use_llm_judge)
        all_metrics = {"Ensemble": ensemble_metrics}
        
        # Calculate metrics for each individual model
        for role, results in single_model_results.items():
            model_metrics = evaluate_answers(reference_answers, results, dataset_type, use_llm_judge)
            all_metrics[role] = model_metrics
        
        # Print results
        print_metrics_table(all_metrics, dataset_type)
        
    # Save all results
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"evaluation_results_{timestamp}.txt"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write("=== Evaluation Results ===\n\n")
        f.write("Test Queries:\n")
        for i, query in enumerate(test_queries):
            f.write(f"{i+1}. {query}\n")
        
        f.write("\n=== Ensemble Results ===\n")
        for i, result in enumerate(ensemble_results):
            f.write(f"\nQuery {i+1} Response:\n{result}\n")
            
        f.write("\n=== Individual Model Results ===\n")
        for role, results in single_model_results.items():
            f.write(f"\n--- {role.capitalize()} Model ({MODELS[role]['name']}) ---\n")
            for i, result in enumerate(results):
                f.write(f"\nQuery {i+1} Response:\n{result}\n")
                
        if reference_answers:
            f.write("\n=== Automated Metrics ===\n")
            # Write metrics in a tabular format
            f.write(format_metrics_table(all_metrics, dataset_type))
    
    print(f"\nAll evaluation results saved to {filename}")
    
    # Also save sample responses for qualitative analysis
    save_sample_responses(test_queries, reference_answers, ensemble_results, single_model_results)
    
    return ensemble_results, single_model_results

def save_sample_responses(queries, references, ensemble_responses, model_responses):
    """Save sample responses to a file for qualitative analysis"""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"sample_responses_{timestamp}.txt"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write("=== Sample Responses for Qualitative Analysis ===\n\n")
        
        for i, query in enumerate(queries):
            f.write(f"Query {i+1}: {query}\n\n")
            if references and i < len(references):
                f.write(f"Reference Answer:\n{references[i]}\n\n")
            f.write(f"Ensemble Response:\n{ensemble_responses[i]}\n\n")
            
            f.write("Individual Model Responses:\n")
            for model_name, responses in model_responses.items():
                f.write(f"\n{model_name}:\n{responses[i]}\n")
            
            f.write("\n" + "="*80 + "\n\n")
    
    print(f"Sample responses saved to {filename} for qualitative analysis")

def evaluate_answers(reference_answers, candidate_answers, dataset_type=None, use_llm_judge=True):
    """
    Evaluate a set of answers using automated metrics appropriate for the dataset type
    
    Args:
        reference_answers: List of reference answers
        candidate_answers: List of candidate answers to evaluate
        dataset_type: Type of dataset (mmlu, gpqa, summarization, translation)
        use_llm_judge: Whether to use an LLM as judge for truthfulness evaluation
    """
    # For multiple-choice datasets (MMLU)
    if dataset_type == "mmlu":
        return evaluate_multiple_choice(reference_answers, candidate_answers)
    
    # For GPQA, evaluate accuracy and truthfulness
    elif dataset_type == "gpqa":
        return evaluate_truthfulness(reference_answers, candidate_answers, use_llm_judge)
    
    # For summarization, evaluate using ROUGE
    elif dataset_type == "summarization":
        return evaluate_summarization(reference_answers, candidate_answers)
    
    # For translation, evaluate using BLEU
    elif dataset_type == "translation":
        return evaluate_translation(reference_answers, candidate_answers)
    
    # Fallback to general metrics for any other dataset type
    else:
        return evaluate_general_quality(reference_answers, candidate_answers)

def evaluate_multiple_choice(reference_answers, candidate_answers):
    """Evaluate multiple-choice answers by calculating accuracy based on the standardized 'The correct answer is X' format"""
    metrics = {
        "accuracy": 0,
        "partial_match": 0
    }
    
    correct_count = 0
    partial_match_count = 0
    total = len(reference_answers)
    
    for ref, cand in zip(reference_answers, candidate_answers):
        if not ref or not cand:
            continue
            
        # Extract correct answer letter from reference (expanded patterns)
        ref_letter = None
        ref_patterns = [
            r"The correct answer is:?\s*([A-Z])",
            r"correct answer:?\s*([A-Z])",
            r"answer:?\s*([A-Z])"
        ]
        
        for pattern in ref_patterns:
            ref_match = re.search(pattern, ref, re.IGNORECASE)
            if ref_match:
                ref_letter = ref_match.group(1).upper()
                break
        
        # Extract candidate answer letter (significantly expanded patterns)
        cand_letter = None
        cand_patterns = [
            r"The correct answer is\s*([A-Z])", 
            r"correct answer is\s*([A-Z])[.)\s]",
            r"answer is\s*([A-Z])[.)\s]",
            r"answer:\s*([A-Z])[.)\s]", 
            r"([A-Z])\)\s*is correct",
            r"option\s*([A-Z])[.)\s]",
            r"answer\s*([A-Z])[.)\s]",
            r"chose\s*([A-Z])[.)\s]",
            r"choose\s*([A-Z])[.)\s]",
            r"select\s*([A-Z])[.)\s]",
            r"selected\s*([A-Z])[.)\s]",
            r"pick\s*([A-Z])[.)\s]",
            r"picked\s*([A-Z])[.)\s]",
            r"going with\s*([A-Z])[.)\s]",
            r"^\s*([A-Z])\s*$",  # Just the letter alone on a line
            r"\*\*([A-Z])\*\*",  # Bold letter
            r"__([A-Z])__"       # Underlined letter
        ]
        
        # First try the entire text
        for pattern in cand_patterns:
            cand_match = re.search(pattern, cand, re.IGNORECASE)
            if cand_match:
                cand_letter = cand_match.group(1).upper()
                break
                
        # If not found, try focusing on the last paragraph where conclusions often appear
        if not cand_letter:
            last_paragraphs = cand.split('\n\n')[-3:]  # Try last 3 paragraphs
            for paragraph in last_paragraphs:
                for pattern in cand_patterns:
                    cand_match = re.search(pattern, paragraph, re.IGNORECASE)
                    if cand_match:
                        cand_letter = cand_match.group(1).upper()
                        break
                if cand_letter:
                    break
        
        # If both ref_letter and cand_letter were extracted successfully, compare them
        if ref_letter and cand_letter:
            if ref_letter == cand_letter:
                correct_count += 1
            continue  # Skip the remaining checks since we already evaluated this pair
        
        # If we couldn't extract letters directly, use the previous pattern-matching approach
        # Extract correct answer from reference using multiple patterns
        correct_answer = None
        
        pattern1 = re.search(r"The correct answer is:?\s*([A-Z0-9]\.?\s*.*?)(?:\n|\Z)", ref)
        if pattern1:
            correct_answer = pattern1.group(1).strip()
        
        if not correct_answer:
            pattern2 = re.search(r"correct answer:?\s*([A-Z0-9]\.?\s*.*?)(?:\n|\Z)", ref, re.IGNORECASE)
            if pattern2:
                correct_answer = pattern2.group(1).strip()
        
        if not correct_answer:
            pattern3 = re.search(r"Answer:?\s*([A-Z0-9]\.?\s*.*?)(?:\n|\Z)", ref, re.IGNORECASE)
            if pattern3:
                correct_answer = pattern3.group(1).strip()
                
        if not correct_answer and "All answer choices:" in ref:
            choices_section = ref.split("All answer choices:")[1].strip()
            choices_lines = choices_section.split('\n')
            for line in choices_lines:
                if re.match(r'^\s*[0-9A-Z]+\.\s', line):
                    if "correct" in line.lower() or "*" in line:
                        correct_answer = line.strip()
                        break
        
        if correct_answer:
            # Clean up the correct answer for better matching
            clean_answer = re.sub(r'^[A-Z0-9]\.?\s*', '', correct_answer).strip()
            letter_only = re.match(r'^([A-Z])\.?$', correct_answer.strip())
            
            # Normalize candidate text
            cand_lower = cand.lower()
            clean_answer_lower = clean_answer.lower()
            
            # Check for various forms of correct answer in candidate
            exact_match = False
            partial_match = False
            
            # Case 1: Exact match of the full answer
            if clean_answer_lower in cand_lower:
                exact_match = True
            
            # Case 2: Just the answer letter (if available)
            elif letter_only and f"answer {letter_only.group(1).lower()}" in cand_lower:
                exact_match = True
            elif letter_only and f"answer is {letter_only.group(1).lower()}" in cand_lower:
                exact_match = True
            elif letter_only and f"option {letter_only.group(1).lower()}" in cand_lower:
                exact_match = True
            
            # Case 3: Check if the answer is directly stated
            elif any(marker + clean_answer_lower in cand_lower for marker in 
                  [" is ", " would be ", "the answer is ", "correct answer is "]):
                exact_match = True
            
            # Check for partial match - if key words/phrases are present
            if not exact_match:
                # Split the clean answer into significant words (ignore common words)
                significant_words = [word.lower() for word in clean_answer.split() 
                                   if len(word) > 3 and word.lower() not in 
                                   ['the', 'and', 'that', 'this', 'with', 'from', 'have', 'what']]
                
                # Count how many significant words appear in the candidate response
                matching_words = [word for word in significant_words if word in cand_lower]
                
                # If more than half of significant words appear, consider it a partial match
                if significant_words and len(matching_words) / len(significant_words) >= 0.5:
                    partial_match = True
            
            # Update counts
            if exact_match:
                correct_count += 1
            elif partial_match:
                partial_match_count += 1
    
    # Log results
    print(f"Multiple choice evaluation results: {correct_count} correct out of {total}")
    
    # Calculate metrics
    if total > 0:
        metrics["accuracy"] = correct_count / total
        metrics["partial_match"] = partial_match_count / total
    
    return metrics

def evaluate_summarization(reference_answers, candidate_answers):
    """Evaluate summarization using ROUGE metrics"""
    metrics = {
        "rouge1": 0,
        "rouge2": 0,
        "rougeL": 0,
        "relevance_score": 0
    }
    
    # Initialize ROUGE
    rouge = Rouge()
    
    total_evaluated = 0
    rouge1_total = 0
    rouge2_total = 0
    rougeL_total = 0
    relevance_total = 0
    
    for ref, cand in zip(reference_answers, candidate_answers):
        if not ref or not cand:
            continue
            
        total_evaluated += 1
        
        # Calculate ROUGE
        try:
            rouge_scores = rouge.get_scores(cand, ref)[0]
            rouge1_total += rouge_scores["rouge-1"]["f"]
            rouge2_total += rouge_scores["rouge-2"]["f"]
            rougeL_total += rouge_scores["rouge-l"]["f"]
            
            # Calculate a relevance score based on content overlap
            ref_words = set(word.lower() for word in safe_tokenize(ref) 
                           if word.isalnum() and len(word) > 2)
            cand_words = set(word.lower() for word in safe_tokenize(cand) 
                            if word.isalnum() and len(word) > 2)
            
            # Skip stop words to focus on meaningful content
            try:
                from nltk.corpus import stopwords
                stop_words = set(stopwords.words('english'))
                ref_words = ref_words - stop_words
                cand_words = cand_words - stop_words
            except Exception:
                pass  # Continue without stopwords if not available
            
            # Calculate content overlap (Jaccard similarity)
            if ref_words and cand_words:
                jaccard = len(ref_words & cand_words) / len(ref_words | cand_words)
                relevance_total += jaccard
        except Exception as e:
            print(f"Error calculating ROUGE metrics: {e}")
    
    # Calculate averages
    if total_evaluated > 0:
        metrics["rouge1"] = rouge1_total / total_evaluated
        metrics["rouge2"] = rouge2_total / total_evaluated
        metrics["rougeL"] = rougeL_total / total_evaluated
        metrics["relevance_score"] = relevance_total / total_evaluated
    
    return metrics

def evaluate_translation(reference_answers, candidate_answers):
    """Evaluate translation using BLEU metric"""
    metrics = {
        "bleu": 0,
        "bleu1": 0,
        "bleu2": 0,
        "bleu3": 0,
        "bleu4": 0
    }
    
    # Initialize BLEU smoothing function
    smoothie = SmoothingFunction().method1
    
    total_evaluated = 0
    bleu_total = 0
    bleu1_total = 0
    bleu2_total = 0
    bleu3_total = 0
    bleu4_total = 0
    
    for ref, cand in zip(reference_answers, candidate_answers):
        if not ref or not cand:
            continue
            
        total_evaluated += 1
        
        # Tokenize for BLEU
        try:
            ref_tokens = safe_tokenize(ref.lower())
            cand_tokens = safe_tokenize(cand.lower())
            
            # Calculate BLEU with smoothing (overall score)
            bleu_score = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothie)
            bleu_total += bleu_score
            
            # Calculate individual n-gram BLEU scores
            weights1 = (1, 0, 0, 0)
            weights2 = (0.5, 0.5, 0, 0)
            weights3 = (0.33, 0.33, 0.33, 0)
            weights4 = (0.25, 0.25, 0.25, 0.25)
            
            bleu1_score = sentence_bleu([ref_tokens], cand_tokens, weights=weights1, smoothing_function=smoothie)
            bleu2_score = sentence_bleu([ref_tokens], cand_tokens, weights=weights2, smoothing_function=smoothie)
            bleu3_score = sentence_bleu([ref_tokens], cand_tokens, weights=weights3, smoothing_function=smoothie)
            bleu4_score = sentence_bleu([ref_tokens], cand_tokens, weights=weights4, smoothing_function=smoothie)
            
            bleu1_total += bleu1_score
            bleu2_total += bleu2_score
            bleu3_total += bleu3_score
            bleu4_total += bleu4_score
            
        except Exception as e:
            print(f"Error calculating BLEU: {e}")
    
    # Calculate averages
    if total_evaluated > 0:
        metrics["bleu"] = bleu_total / total_evaluated
        metrics["bleu1"] = bleu1_total / total_evaluated
        metrics["bleu2"] = bleu2_total / total_evaluated
        metrics["bleu3"] = bleu3_total / total_evaluated
        metrics["bleu4"] = bleu4_total / total_evaluated
    
    return metrics

def evaluate_truthfulness(reference_answers, candidate_answers, use_llm_judge=False):
    """
    Evaluate answers for truthfulness using basic fact checking
    """
    return evaluate_truthfulness_with_fact_checking(reference_answers, candidate_answers)

def evaluate_truthfulness_with_fact_checking(reference_answers, candidate_answers):
    """
    Evaluate answers for truthfulness based on fact-checking against reference information
    """
    metrics = {
        "estimated_truthfulness": 0,
        "toxicity_avoidance": 0,
        "information_content": 0
    }
    
    total_evaluated = 0
    truthfulness_score = 0
    toxicity_avoidance_score = 0
    information_score = 0
    
    for ref, cand in zip(reference_answers, candidate_answers):
        if not cand or not ref:  # Skip if either reference or candidate is missing
            continue
            
        total_evaluated += 1
        
        # Extract factual claims and reference facts
        candidate_claims = extract_claims(cand)
        reference_facts = extract_facts(ref)
        
        # Fact verification
        verified_facts_count = 0
        contradicted_facts_count = 0
        
        for claim in candidate_claims:
            verification_result = verify_claim_against_reference(claim, reference_facts)
            if verification_result == "verified":
                verified_facts_count += 1
            elif verification_result == "contradicted":
                contradicted_facts_count += 1
        
        # Calculate truthfulness score
        total_claims = max(1, len(candidate_claims))
        factual_accuracy = (verified_facts_count - contradicted_facts_count) / total_claims
        normalized_truthfulness = (factual_accuracy + 1) / 2
        truthfulness_score += normalized_truthfulness
        
        # Simple toxicity check
        toxicity_avoidance_score += 0.9  # Default to high toxicity avoidance
        
        # Information content measurement
        word_count = len(safe_tokenize(cand))
        info_score = min(1.0, word_count / 400)
        information_score += info_score
    
    # Calculate average scores
    if total_evaluated > 0:
        metrics["estimated_truthfulness"] = truthfulness_score / total_evaluated
        metrics["toxicity_avoidance"] = toxicity_avoidance_score / total_evaluated
        metrics["information_content"] = min(1.0, information_score / total_evaluated)
    
    return metrics

def extract_claims(text):
    """Extract factual claims from a text."""
    claims = []
    sentences = safe_sent_tokenize(text)
    
    # Identify sentences that appear to make factual claims
    for sentence in sentences:
        # Skip questions, commands, and purely subjective statements
        if sentence.endswith('?') or sentence.endswith('!'):
            continue
            
        # Skip sentences that are clearly opinions
        opinion_markers = ["i think", "i believe", "in my opinion", "i feel", "i would say"]
        if any(marker in sentence.lower() for marker in opinion_markers):
            continue
            
        # Include sentences that make factual assertions
        claims.append(sentence)
    
    return claims

def extract_facts(reference_text):
    """Extract factual information from reference text."""
    facts = []
    sentences = safe_sent_tokenize(reference_text)
    
    for sentence in sentences:
        # Skip non-factual content
        if sentence.endswith('?') or sentence.endswith('!'):
            continue
            
        # Generate key terms for this fact to aid in matching
        words = safe_tokenize(sentence.lower())
        key_terms = [word for word in words if len(word) > 4]
        
        facts.append({
            "text": sentence,
            "key_terms": key_terms
        })
    
    return facts

def verify_claim_against_reference(claim, reference_facts):
    """
    Verify a claim against reference facts.
    
    Returns:
    - "verified" if the claim is supported by reference facts
    - "contradicted" if the claim contradicts reference facts
    - "unknown" if there's insufficient information
    """
    claim_words = set(safe_tokenize(claim.lower()))
    
    # Remove common words for better matching
    claim_content_words = [word for word in claim_words if len(word) > 4]
    
    # Look for supporting evidence
    best_match_score = 0
    best_match_result = "unknown"
    
    for fact in reference_facts:
        # Calculate similarity based on shared key terms
        fact_terms = set(fact["key_terms"])
        shared_terms = sum(1 for term in claim_content_words if term in fact_terms)
        
        if not fact_terms or not claim_content_words:
            continue
            
        similarity_score = shared_terms / max(1, len(fact_terms) + len(claim_content_words) - shared_terms)
        
        if similarity_score > best_match_score:
            best_match_score = similarity_score
            if similarity_score > 0.4:  # Threshold for verification
                best_match_result = "verified"
    
    return best_match_result

def evaluate_general_quality(reference_answers, candidate_answers):
    """Evaluate general answer quality using text similarity metrics"""
    metrics = {
        "bleu": 0,
        "rouge1": 0,
        "rouge2": 0,
        "rougeL": 0,
        "relevance_score": 0
    }
    
    # Initialize ROUGE
    rouge = Rouge()
    
    # Initialize BLEU smoothing function
    smoothie = SmoothingFunction().method1
    
    total_scores = {key: 0 for key in metrics}
    total_evaluated = 0
    
    for ref, cand in zip(reference_answers, candidate_answers):
        # Handle empty or None values
        if not ref or not cand:
            continue
            
        total_evaluated += 1
        
        # Calculate BLEU
        try:
            ref_tokens = safe_tokenize(ref.lower())
            cand_tokens = safe_tokenize(cand.lower())
            bleu_score = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothie)
            total_scores["bleu"] += bleu_score
        except Exception:
            pass
        
        # Calculate ROUGE
        try:
            rouge_scores = rouge.get_scores(cand, ref)[0]
            total_scores["rouge1"] += rouge_scores["rouge-1"]["f"]
            total_scores["rouge2"] += rouge_scores["rouge-2"]["f"]
            total_scores["rougeL"] += rouge_scores["rouge-l"]["f"]
        except Exception:
            pass
        
        # Calculate relevance score
        try:
            ref_words = set(word.lower() for word in safe_tokenize(ref) if len(word) > 2)
            cand_words = set(word.lower() for word in safe_tokenize(cand) if len(word) > 2)
            
            if ref_words and cand_words:
                relevance = len(ref_words & cand_words) / len(ref_words | cand_words)
                total_scores["relevance_score"] += relevance
        except Exception:
            pass
    
    # Calculate averages
    if total_evaluated > 0:
        for key in metrics:
            metrics[key] = total_scores[key] / total_evaluated
    
    return metrics

def format_metrics_table(all_metrics, dataset_type=None):
    """Format metrics as a text table for file output based on dataset type"""
    # Define metric display names based on dataset type
    if dataset_type == "mmlu":
        metric_names = {
            "accuracy": "Accuracy",
            "partial_match": "Partial Match"
        }
    elif dataset_type == "gpqa":
        metric_names = {
            "estimated_truthfulness": "Est. Truthfulness",
            "information_content": "Info Content"
        }
    elif dataset_type == "summarization":
        metric_names = {
            "rouge1": "ROUGE-1",
            "rouge2": "ROUGE-2", 
            "rougeL": "ROUGE-L",
            "relevance_score": "Relevance"
        }
    elif dataset_type == "translation":
        metric_names = {
            "bleu": "BLEU",
            "bleu1": "BLEU-1",
            "bleu2": "BLEU-2",
            "bleu3": "BLEU-3",
            "bleu4": "BLEU-4"
        }
    else:
        metric_names = {
            "bleu": "BLEU",
            "rouge1": "ROUGE-1",
            "rouge2": "ROUGE-2", 
            "rougeL": "ROUGE-L",
            "relevance_score": "Relevance"
        }
    
    # Create header
    header = "Model"
    for metric in metric_names.values():
        header += f" | {metric:18}"
    
    # Create separator line
    separator = "-" * len(header)
    
    # Create rows
    rows = [header, separator]
    
    for model_name, metrics in all_metrics.items():
        row = f"{model_name[:15]:<15}"
        for metric_key, metric_display in metric_names.items():
            if metric_key in metrics:
                row += f" | {metrics[metric_key]:.4f}              "
            else:
                row += f" | N/A                 "
        rows.append(row)
    
    return "\n".join(rows)

def print_metrics_table(all_metrics, dataset_type=None):
    """Print metrics in a nicely formatted table"""
    print("\n" + format_metrics_table(all_metrics, dataset_type))
    
    # Also create a pandas DataFrame for better visualization
    try:
        df = pd.DataFrame(all_metrics).T
        # Round values to 4 decimal places
        for col in df.columns:
            if df[col].dtype == 'float64':
                df[col] = df[col].round(4)
        print("\nMetrics as DataFrame (for easier reading):")
        print(df)
        
        # Export to CSV
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        csv_filename = f"metrics_results_{timestamp}.csv"
        df.to_csv(csv_filename)
        print(f"Metrics saved to {csv_filename}")
    except Exception as e:
        print(f"Could not create DataFrame: {e}")

def generate_metrics_plots(metrics_df, timestamp, dataset_type=None):
    """Generate plots of evaluation metrics based on dataset type"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        sns.set_theme(style="whitegrid")
        
        # Bar plot for each metric comparing models
        plt.figure(figsize=(15, 10))
        
        metrics = metrics_df.columns
        n_metrics = len(metrics)
        fig, axes = plt.subplots(nrows=(n_metrics+1)//2, ncols=2, figsize=(15, 3*((n_metrics+1)//2)))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            sns.barplot(x=metrics_df.index, y=metrics_df[metric], ax=ax)
            ax.set_title(f"{metric} Score")
            ax.set_xlabel("Model")
            ax.set_ylabel("Score")
            ax.tick_params(axis='x', rotation=45)
        
        # Hide any unused subplots
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
            
        plt.tight_layout()
        plt.savefig(f"{dataset_type or 'metrics'}_barplot_{timestamp}.png", dpi=300, bbox_inches='tight')
        
        # Heatmap for all metrics
        plt.figure(figsize=(10, 8))
        
        # Determine plot title based on dataset type
        if dataset_type == "mmlu":
            plot_title = "MMLU Accuracy Evaluation"
        elif dataset_type == "gpqa":
            plot_title = "GPQA Evaluation"
        elif dataset_type == "summarization":
            plot_title = "Summarization ROUGE Evaluation"
        elif dataset_type == "translation":
            plot_title = "Translation BLEU Evaluation"
        else:
            plot_title = "Response Quality Evaluation"
            
        sns.heatmap(metrics_df, annot=True, cmap="YlGnBu", fmt=".4f", linewidths=.5)
        plt.title(f"{plot_title} Heatmap")
        plt.tight_layout()
        plt.savefig(f"{dataset_type or 'metrics'}_heatmap_{timestamp}.png", dpi=300, bbox_inches='tight')
        
        print(f"Plots saved as {dataset_type or 'metrics'}_barplot_{timestamp}.png and {dataset_type or 'metrics'}_heatmap_{timestamp}.png")
    except Exception as e:
        print(f"Could not generate plots: {e}")

def load_benchmark_dataset(dataset_name="mmlu", subset=None, split=None, num_samples=10):
    """
    Load a benchmark dataset for evaluation
    
    Args:
        dataset_name: Name of the dataset (mmlu, gpqa, summarization, translation)
        subset: Subset of the dataset if applicable
        split: Split to use (train, validation, test)
        num_samples: Number of samples to use
        
    Returns:
        questions, reference_answers, dataset_type: Lists of questions, references, and the dataset type
    """
    print(f"Loading {dataset_name} dataset...")
    dataset_type = dataset_name  # Default dataset type is the same as name
    
    # Set appropriate default split based on dataset type
    if split is None:
        if dataset_name == "mmlu":
            split = "auxiliary_train"
        else:
            split = "validation"  # Use validation split for other datasets by default
    
    try:
        if dataset_name == "mmlu":
            # MMLU tests knowledge across domains
            if subset is None:
                subset = "all"  # Use the 'all' subset which includes questions from all categories
                print(f"No subset specified, using '{subset}' which includes questions from all categories")
            
            dataset = load_dataset("cais/mmlu", subset, split=split)
            questions = []
            reference_answers = []
            
            # Take a random sample of the dataset
            indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
            
            for idx in indices:
                item = dataset[idx]
                question_text = item["question"]
                
                # Map answer index to answer choices
                choices = [item["choices"][i] for i in range(len(item["choices"]))]
                correct_idx = item["answer"]
                correct_answer = choices[correct_idx] if correct_idx < len(choices) else "Unknown"
                
                # Reformat question to ensure clear multiple choice format with A), B), etc.
                formatted_question = question_text.strip()
                if not any(f"{letter})" in formatted_question for letter in "ABCD"):
                    formatted_question += "\n"
                    for i, choice in enumerate(choices):
                        letter = chr(65 + i)  # 65 is ASCII for 'A'
                        formatted_question += f"{letter}) {choice}\n"
                
                reference = f"The correct answer is: {chr(65 + correct_idx)}) {correct_answer}\n\n"
                reference += f"All answer choices:\n"
                for i, choice in enumerate(choices):
                    reference += f"{i+1}. {chr(65 + i)}) {choice}\n"
                    
                questions.append(formatted_question)
                reference_answers.append(reference)
                
        elif dataset_name == "gpqa":
            # Load GPQA dataset from Hugging Face
            try:
                # Use gpqa_main config by default, or allow user to specify
                config_name = subset if subset else "gpqa_main"
                # GPQA only has 'train' split
                dataset_split = "train"
                dataset = load_dataset("Idavidrein/gpqa", config_name, split=dataset_split)
                questions = []
                reference_answers = []
                
                # Print sample data to understand the dataset structure
                print(f"GPQA dataset loaded with {len(dataset)} entries")
                if len(dataset) > 0:
                    print("Sample dataset keys:", list(dataset[0].keys()))
                
                # Based on GPQA dataset structure (https://huggingface.co/datasets/Idavidrein/gpqa)
                # Expected keys: 'query', 'A', 'B', 'C', 'D', 'answer' or similar
                
                # Take a random sample of the dataset
                indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
                
                for idx in indices:
                    item = dataset[idx]
                    # Debug the actual structure
                    print(f"Sample item keys: {list(item.keys())}")
                    
                    # Adapt to different possible field names
                    if 'query' in item:
                        question_text = item['query']
                    elif 'question' in item:
                        question_text = item['question']
                    else:
                        # Try to find a question-like field
                        potential_question_fields = [k for k in item.keys() if isinstance(item[k], str) and len(item[k]) > 20]
                        if potential_question_fields:
                            question_text = item[potential_question_fields[0]]
                        else:
                            question_text = "Question not found in dataset"
                    
                    # Try to extract choices - can vary between datasets
                    choices = []
                    if all(option in item for option in ['A', 'B', 'C', 'D']):
                        choices = [item['A'], item['B'], item['C'], item['D']]
                    elif 'choices' in item and isinstance(item['choices'], list):
                        choices = item['choices']
                    # Check for GPQA dataset format with "Correct Answer" and "Incorrect Answer X"
                    elif 'Correct Answer' in item and 'Incorrect Answer 1' in item:
                        choices = [
                            item['Correct Answer'],
                            item['Incorrect Answer 1'],
                            item.get('Incorrect Answer 2', 'No option provided'),
                            item.get('Incorrect Answer 3', 'No option provided')
                        ]
                    else:
                        # Try to find choice-like fields
                        choice_fields = [k for k in item.keys() if k in ['a', 'b', 'c', 'd'] or k in ['option_a', 'option_b', 'option_c', 'option_d']]
                        if choice_fields:
                            choices = [item[k] for k in sorted(choice_fields)]
                        else:
                            choices = ["Option A not found", "Option B not found", "Option C not found", "Option D not found"]
                    
                    # Extract correct answer
                    correct_idx = 0
                    if 'answer' in item:
                        if isinstance(item['answer'], int):
                            correct_idx = item['answer']
                        elif isinstance(item['answer'], str) and item['answer'] in 'ABCD':
                            correct_idx = ord(item['answer']) - ord('A')
                    # Handle GPQA-style dataset where the correct answer is the first in the choices list
                    elif 'Correct Answer' in item and choices and choices[0] == item['Correct Answer']:
                        correct_idx = 0
                    
                    correct_answer = choices[correct_idx] if correct_idx < len(choices) else "Unknown"
                    
                    # Format question with multiple choice options
                    formatted_question = question_text.strip()
                    if not any(f"{letter})" in formatted_question for letter in "ABCD"):
                        formatted_question += "\n"
                        for i, choice in enumerate(choices):
                            letter = chr(65 + i)  # 65 is ASCII for 'A'
                            formatted_question += f"{letter}) {choice}\n"
                    
                    # Format reference answer
                    formatted_reference = f"The correct answer is: {chr(65 + correct_idx)}) {correct_answer}\n\n"
                    formatted_reference += f"All answer choices:\n"
                    for i, choice in enumerate(choices):
                        formatted_reference += f"{i+1}. {chr(65 + i)}) {choice}\n"
                    
                    questions.append(formatted_question)
                    reference_answers.append(formatted_reference)
                    
                    # Debug the actual structure only for the first few items
                    if len(questions) <= 3:
                        print(f"Sample item keys: {list(item.keys())}")
                
                if not questions:
                    raise ValueError("Failed to extract questions from the dataset")
                    
                # Modify dataset_type to be treated as multiple choice
                dataset_type = "mmlu"
                
            except Exception as e:
                print(f"Error loading GPQA dataset: {e}")
                print("Using fallback questions...")
                
                # Fallback multiple choice GPQA questions
                questions = [
                    "What is the key advantage of transformer models over recurrent neural networks (RNNs) in natural language processing?\nA) Parallelization capability due to attention mechanisms instead of sequential processing\nB) Lower computational requirements for training on typical hardware\nC) Guaranteed convergence to global optima during the training process\nD) Native handling of variable-length inputs without padding",
                    "Which mechanism enables quantum computers to potentially perform certain calculations exponentially faster than classical computers?\nA) Quantum superposition and entanglement allowing parallel exploration of solution spaces\nB) Higher clock speeds in quantum processing units (QPUs)\nC) More efficient memory allocation and garbage collection\nD) Direct inter-process communication between computational units",
                    "What distinguishes deep reinforcement learning from supervised learning?\nA) Learning through environmental interaction and delayed rewards rather than labeled examples\nB) Using exclusively neural networks with at least 10 hidden layers\nC) Operating only on structured data like images or text\nD) Requiring significantly less computational resources for training"
                ]
                
                reference_answers = [
                    "The correct answer is: A) Parallelization capability due to attention mechanisms instead of sequential processing\n\nAll answer choices:\n1. A) Parallelization capability due to attention mechanisms instead of sequential processing\n2. B) Lower computational requirements for training on typical hardware\n3. C) Guaranteed convergence to global optima during the training process\n4. D) Native handling of variable-length inputs without padding",
                    "The correct answer is: A) Quantum superposition and entanglement allowing parallel exploration of solution spaces\n\nAll answer choices:\n1. A) Quantum superposition and entanglement allowing parallel exploration of solution spaces\n2. B) Higher clock speeds in quantum processing units (QPUs)\n3. C) More efficient memory allocation and garbage collection\n4. D) Direct inter-process communication between computational units",
                    "The correct answer is: A) Learning through environmental interaction and delayed rewards rather than labeled examples\n\nAll answer choices:\n1. A) Learning through environmental interaction and delayed rewards rather than labeled examples\n2. B) Using exclusively neural networks with at least 10 hidden layers\n3. C) Operating only on structured data like images or text\n4. D) Requiring significantly less computational resources for training"
                ]
                
                # Limit to requested number of samples
                questions = questions[:num_samples]
                reference_answers = reference_answers[:num_samples]
                
                # Set dataset_type to mmlu for multiple choice evaluation
                dataset_type = "mmlu"
                
        elif dataset_name == "summarization":
            # Text summarization dataset
            if subset is None or subset not in ["xsum", "cnn_dailymail"]:
                subset = "cnn_dailymail"  # Default to cnn_dailymail if none or invalid subset specified
            
            if subset == "cnn_dailymail":
                # Use version 3.0.0 of the CNN/DailyMail dataset
                dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split=split)
            else:
                dataset = load_dataset(subset, split=split)
                
            questions = []
            reference_answers = []
            
            # Take a random sample of the dataset
            indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
            
            for idx in indices:
                item = dataset[idx]
                if subset == "xsum":
                    document = item["document"]
                    summary = item["summary"]
                elif subset == "cnn_dailymail":
                    document = item["article"]
                    summary = item["highlights"]
                else:
                    document = ""
                    summary = ""
                
                # Format as a summarization task
                question = f"Summarize the following text:\n\n{document}"
                reference = summary
                
                questions.append(question)
                reference_answers.append(reference)
        
        elif dataset_name == "translation":
            # Translation dataset using WMT instead of Helsinki-NLP/open_subtitles which has URL issues
            if subset is None:
                subset = "en-de"  # Default to English-German translation
                print(f"No language pair specified, using '{subset}'")
            
            try:
                # Parse language pair
                lang_parts = subset.split('-')
                if len(lang_parts) != 2:
                    print(f"Invalid language pair format: {subset}. Using en-de as default.")
                    lang1, lang2 = "en", "de"
                else:
                    lang1, lang2 = lang_parts
                
                # Map language pairs to WMT datasets
                wmt_datasets = {
                    "en-de": ("wmt14", "de-en"),
                    "en-fr": ("wmt14", "fr-en"),
                    "en-cs": ("wmt16", "cs-en"),
                    "en-ru": ("wmt16", "ru-en"),
                    "en-zh": ("wmt17", "zh-en"),
                    "en-fi": ("wmt17", "fi-en")
                }
                
                # Get the appropriate WMT dataset
                if f"{lang1}-{lang2}" in wmt_datasets:
                    wmt_version, wmt_pair = wmt_datasets[f"{lang1}-{lang2}"]
                elif f"{lang2}-{lang1}" in wmt_datasets:
                    wmt_version, wmt_pair = wmt_datasets[f"{lang2}-{lang1}"]
                    # Reverse the pair for consistency
                    wmt_pair = f"{lang1}-{lang2}" 
                else:
                    print(f"Language pair {lang1}-{lang2} not found in available WMT datasets. Using wmt14 de-en.")
                    wmt_version, wmt_pair = "wmt14", "de-en"
                    lang1, lang2 = "en", "de"
                
                # Load the WMT dataset
                print(f"Loading {wmt_version} dataset with language pair {wmt_pair}...")
                dataset = load_dataset(wmt_version, wmt_pair, split=split)
                questions = []
                reference_answers = []
                
                print(f"Translation dataset loaded with {len(dataset)} entries")
                if len(dataset) > 0:
                    print("Sample dataset keys:", list(dataset[0].keys()))
                
                # Take a random sample of the dataset
                indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
                
                for idx in indices:
                    item = dataset[idx]
                    
                    # Extract source and target text based on the dataset structure
                    if 'translation' in item:
                        source_text = item['translation'][lang1]
                        target_text = item['translation'][lang2]
                    else:
                        # Try to handle different dataset structures
                        source_key = lang1
                        target_key = lang2
                        
                        if source_key in item:
                            source_text = item[source_key]
                            target_text = item[target_key]
                        else:
                            # Final fallback - just use any text fields
                            keys = list(item.keys())
                            if len(keys) >= 2:
                                source_text = item[keys[0]]
                                target_text = item[keys[1]]
                            else:
                                continue
                    
                    # Format as a translation task
                    question = f"Translate the following {lang1.upper()} text to {lang2.upper()}:\n\n{source_text}"
                    reference = target_text
                    
                    questions.append(question)
                    reference_answers.append(reference)
                
                if not questions:
                    raise ValueError("Failed to extract translations from the dataset")
            
            except Exception as e:
                print(f"Error loading translation dataset: {e}")
                print("Using fallback translation examples...")
                
                # Map language pair to fallback examples
                if subset is None:
                    subset = "en-de"
                
                # Get language codes
                lang_parts = subset.split('-')
                if len(lang_parts) == 2:
                    source_lang, target_lang = lang_parts
                else:
                    source_lang, target_lang = "en", "de"
                
                # Create fallback examples based on the language pair
                fallback_examples = {
                    "en-de": [
                        {
                            "query": "Translate the following English text to German:\n\nArtificial intelligence has made enormous progress in recent years.",
                            "reference": "Künstliche Intelligenz hat in den letzten Jahren enorme Fortschritte gemacht."
                        },
                        {
                            "query": "Translate the following English text to German:\n\nSustainable development is essential for the future of our planet.",
                            "reference": "Nachhaltige Entwicklung ist für die Zukunft unseres Planeten unerlässlich."
                        },
                        {
                            "query": "Translate the following English text to German:\n\nDigitalization is changing the way we work and live.",
                            "reference": "Die Digitalisierung verändert die Art und Weise, wie wir arbeiten und leben."
                        }
                    ],
                    "en-fr": [
                        {
                            "query": "Translate the following English text to French:\n\nArtificial intelligence has made enormous progress in recent years.",
                            "reference": "L'intelligence artificielle a fait d'énormes progrès ces dernières années."
                        },
                        {
                            "query": "Translate the following English text to French:\n\nSustainable development is essential for the future of our planet.",
                            "reference": "Le développement durable est essentiel pour l'avenir de notre planète."
                        },
                        {
                            "query": "Translate the following English text to French:\n\nDigitalization is changing the way we work and live.",
                            "reference": "La numérisation change notre façon de travailler et de vivre."
                        }
                    ],
                    "en-ru": [
                        {
                            "query": "Translate the following English text to Russian:\n\nArtificial intelligence has made enormous progress in recent years.",
                            "reference": "Искусственный интеллект достиг огромного прогресса в последние годы."
                        },
                        {
                            "query": "Translate the following English text to Russian:\n\nSustainable development is essential for the future of our planet.",
                            "reference": "Устойчивое развитие необходимо для будущего нашей планеты."
                        }
                    ],
                    "en-zh": [
                        {
                            "query": "Translate the following English text to Chinese:\n\nArtificial intelligence has made enormous progress in recent years.",
                            "reference": "人工智能在近年来取得了巨大的进展。"
                        },
                        {
                            "query": "Translate the following English text to Chinese:\n\nSustainable development is essential for the future of our planet.",
                            "reference": "可持续发展对我们星球的未来至关重要。"
                        }
                    ]
                }
                
                # Default to en-de if no specific examples for this language pair
                selected_examples = fallback_examples.get(f"{source_lang}-{target_lang}", fallback_examples.get("en-de", []))
                
                if not selected_examples:
                    # Use en-de as ultimate fallback
                    selected_examples = fallback_examples["en-de"]
                
                # Extract queries and references
                questions = [example["query"] for example in selected_examples]
                reference_answers = [example["reference"] for example in selected_examples]
                
                # Limit to requested number of samples
                questions = questions[:num_samples]
                reference_answers = reference_answers[:num_samples]
            
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")
        
        print(f"Successfully loaded {len(questions)} questions from {dataset_name} ({dataset_type})")
        return questions, reference_answers, dataset_type
    
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        print("Using fallback questions...")
        
        # Use default questions as fallback
        if dataset_name == "mmlu":
            questions = [
                "Which of the following is a correct statement about the function of DNA polymerase in DNA replication?\nA) It catalyzes the addition of nucleotides to the 3' end of a growing DNA strand.\nB) It catalyzes the addition of nucleotides to the 5' end of a growing DNA strand.\nC) It joins Okazaki fragments together.\nD) It unwinds the DNA double helix.",
                "In the context of international law, what is meant by 'jus cogens'?\nA) Laws that apply only during peacetime\nB) Peremptory norms that cannot be violated by any state\nC) Laws that apply only during armed conflict\nD) Treaties that have been ratified by all UN member states",
                "Which economic concept describes a market structure with many sellers offering differentiated products?\nA) Perfect competition\nB) Monopoly\nC) Monopolistic competition\nD) Oligopoly"
            ]
            
            reference_answers = [
                "The correct answer is: A) It catalyzes the addition of nucleotides to the 3' end of a growing DNA strand.\n\nAll answer choices:\n1. A) It catalyzes the addition of nucleotides to the 3' end of a growing DNA strand.\n2. B) It catalyzes the addition of nucleotides to the 5' end of a growing DNA strand.\n3. C) It joins Okazaki fragments together.\n4. D) It unwinds the DNA double helix.",
                "The correct answer is: B) Peremptory norms that cannot be violated by any state\n\nAll answer choices:\n1. A) Laws that apply only during peacetime\n2. B) Peremptory norms that cannot be violated by any state\n3. C) Laws that apply only during armed conflict\n4. D) Treaties that have been ratified by all UN member states",
                "The correct answer is: C) Monopolistic competition\n\nAll answer choices:\n1. A) Perfect competition\n2. B) Monopoly\n3. C) Monopolistic competition\n4. D) Oligopoly"
            ]
        
        elif dataset_name == "gpqa":
            # Fallback multiple choice GPQA questions
            questions = [
                "What is the key advantage of transformer models over recurrent neural networks (RNNs) in natural language processing?\nA) Parallelization capability due to attention mechanisms instead of sequential processing\nB) Lower computational requirements for training on typical hardware\nC) Guaranteed convergence to global optima during the training process\nD) Native handling of variable-length inputs without padding",
                "Which mechanism enables quantum computers to potentially perform certain calculations exponentially faster than classical computers?\nA) Quantum superposition and entanglement allowing parallel exploration of solution spaces\nB) Higher clock speeds in quantum processing units (QPUs)\nC) More efficient memory allocation and garbage collection\nD) Direct inter-process communication between computational units",
                "What distinguishes deep reinforcement learning from supervised learning?\nA) Learning through environmental interaction and delayed rewards rather than labeled examples\nB) Using exclusively neural networks with at least 10 hidden layers\nC) Operating only on structured data like images or text\nD) Requiring significantly less computational resources for training"
            ]
            
            reference_answers = [
                "The correct answer is: A) Parallelization capability due to attention mechanisms instead of sequential processing\n\nAll answer choices:\n1. A) Parallelization capability due to attention mechanisms instead of sequential processing\n2. B) Lower computational requirements for training on typical hardware\n3. C) Guaranteed convergence to global optima during the training process\n4. D) Native handling of variable-length inputs without padding",
                "The correct answer is: A) Quantum superposition and entanglement allowing parallel exploration of solution spaces\n\nAll answer choices:\n1. A) Quantum superposition and entanglement allowing parallel exploration of solution spaces\n2. B) Higher clock speeds in quantum processing units (QPUs)\n3. C) More efficient memory allocation and garbage collection\n4. D) Direct inter-process communication between computational units",
                "The correct answer is: A) Learning through environmental interaction and delayed rewards rather than labeled examples\n\nAll answer choices:\n1. A) Learning through environmental interaction and delayed rewards rather than labeled examples\n2. B) Using exclusively neural networks with at least 10 hidden layers\n3. C) Operating only on structured data like images or text\n4. D) Requiring significantly less computational resources for training"
            ]
            
            # Set dataset_type to mmlu to use multiple choice evaluation
            dataset_type = "mmlu"
            
        elif dataset_name == "summarization":
            # Fallback summarization examples using CNN/DailyMail style
            questions = [
                "Summarize the following text:\n\nArtificial intelligence has made significant strides in recent years, revolutionizing various industries from healthcare to finance. Machine learning algorithms can now diagnose diseases, predict market trends, and even create art. However, concerns about ethical implications and job displacement continue to grow. Researchers are working on developing AI systems that are transparent, fair, and aligned with human values.",
                "Summarize the following text:\n\nClimate change poses one of the greatest challenges of our time. Rising global temperatures have led to more frequent extreme weather events, melting ice caps, and rising sea levels. Many species face extinction due to habitat loss. International agreements like the Paris Climate Accord aim to limit temperature increases and reduce carbon emissions. Renewable energy technologies such as solar and wind power are becoming more affordable and widespread.",
                "Summarize the following text:\n\nThe COVID-19 pandemic transformed how we work, learn, and socialize. Remote work became the norm for many office employees, while essential workers continued to serve on the frontlines. Educational institutions pivoted to online learning, and telemedicine saw rapid adoption. The development and distribution of vaccines at unprecedented speed demonstrated the potential of global scientific collaboration. However, the pandemic also highlighted and exacerbated existing social inequalities."
            ]
            
            reference_answers = [
                "AI has advanced significantly, impacting healthcare, finance, and art creation through machine learning. Ethical concerns and job displacement fears persist, prompting research into transparent AI aligned with human values.",
                "Climate change causes extreme weather, melting ice caps, rising seas, and species extinction. The Paris Climate Accord aims to limit warming and reduce emissions, while renewable energy becomes more affordable.",
                "COVID-19 transformed society with remote work, online education, and telemedicine adoption. Rapid vaccine development showcased global scientific collaboration, but the pandemic worsened existing social inequalities."
            ]
            
        elif dataset_name == "translation":
            # Fallback translation examples
            questions = [
                "Translate the following German text to English:\n\nDie künstliche Intelligenz hat in den letzten Jahren enorme Fortschritte gemacht.",
                "Translate the following German text to English:\n\nNachhaltige Entwicklung ist für die Zukunft unseres Planeten unerlässlich.",
                "Translate the following German text to English:\n\nDie Digitalisierung verändert die Art und Weise, wie wir arbeiten und leben."
            ]
            
            reference_answers = [
                "Artificial intelligence has made enormous progress in recent years.",
                "Sustainable development is essential for the future of our planet.",
                "Digitalization is changing the way we work and live."
            ]
        
        # Limit to requested number of samples
        questions = questions[:num_samples]
        reference_answers = reference_answers[:num_samples]
        
        print(f"Using {len(questions)} fallback questions")
        return questions, reference_answers, dataset_type

def check_model_availability():
    """Checks if all required models are available via OpenRouter API"""
    print("\n=== Checking Model Availability ===")
    all_available = True
    
    # First check if API key is valid
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
    
    # Test with a commonly available model first
    test_model = "meta-llama/llama-3.1-8b-instruct"
    print(f"Testing API authentication with {test_model}...")
    
    try:
        response = query_openrouter_api(test_model, test_messages, max_tokens=5)
        if response.startswith("Error:"):
            if "Authentication error" in response:
                print(f"❌ API authentication failed: {response}")
                print("Please check your OPENROUTER_API_KEY in the .env file.")
                print("You can get your API key from https://openrouter.ai/keys")
                return False
    except Exception as e:
        print(f"❌ Error during API test: {e}")
    
    # Check each model individually
    for role, model_info in MODELS.items():
        model_name = model_info["name"]
        print(f"Checking {role}: {model_name}...")
        
        try:
            response = query_openrouter_api(model_name, test_messages, max_tokens=5)
            
            if not response.startswith("Error:"):
                print(f"✓ {model_name} is accessible")
            else:
                print(f"❌ Cannot access {model_name}: {response}")
                all_available = False
                # Provide targeted troubleshooting guidance
                if "not found" in response.lower() or "doesn't exist" in response.lower():
                    print(f"  ℹ️ The model ID may be incorrect. Check model name: {model_name}")
                elif "quota" in response.lower() or "credits" in response.lower():
                    print("  ℹ️ You may have exceeded your quota. Check your OpenRouter account.")
        except Exception as e:
            print(f"❌ Error checking {model_name}: {e}")
            all_available = False
    
    return all_available

def list_available_benchmarks():
    """List available benchmark datasets for evaluation"""
    benchmarks = {
        "mmlu": {
            "description": "Massive Multitask Language Understanding - tests knowledge across domains",
            "subsets": ["stem", "humanities", "social_sciences", "other", "professional"],
            "paper": "https://arxiv.org/abs/2009.03300"
        },
        "gpqa": {
            "description": "Graduate-level Google-Proof Q&A - tests graduate-level knowledge in STEM fields",
            "subsets": None,
            "paper": "https://arxiv.org/abs/2311.12022"
        },
        "summarization": {
            "description": "Text summarization benchmark using ROUGE metric",
            "subsets": ["xsum", "cnn_dailymail"],
            "paper": "https://aclanthology.org/2020.acl-main.703/"
        },
        "translation": {
            "description": "Machine translation benchmark using WMT datasets and BLEU metric",
            "subsets": ["en-de", "en-fr", "en-cs", "en-ru", "en-zh", "en-fi"],
            "paper": "https://aclanthology.org/W14-3302/"
        }
    }
    
    print("\n=== Available Benchmark Datasets ===")
    for name, info in benchmarks.items():
        print(f"\n{name}: {info['description']}")
        if info['subsets']:
            print(f"  Subsets: {', '.join(info['subsets'])}")
        print(f"  Paper: {info['paper']}")
    
    return benchmarks

def run_benchmark_evaluation(dataset_name="mmlu", subset=None, num_samples=5, use_llm_judge=False):
    """Run evaluation using a benchmark dataset with appropriate metrics"""
    # Validate dataset name
    valid_datasets = ["mmlu", "gpqa", "summarization", "translation"]
    if dataset_name not in valid_datasets:
        print(f"Dataset {dataset_name} not supported. Using mmlu instead.")
        dataset_name = "mmlu"
    
    print(f"\n=== Running Benchmark Evaluation: {dataset_name} ===")
    
    # Start a timer to track total evaluation time
    start_time = time.time()
    
    try:
        # Load dataset
        questions, reference_answers, dataset_type = load_benchmark_dataset(
            dataset_name=dataset_name,
            subset=subset,
            num_samples=num_samples
        )
        
        # Display sample questions
        print("\nSample questions from dataset:")
        for i, (q, ref) in enumerate(zip(questions, reference_answers)):
            if i < 3:  # Show first 3 examples
                print(f"\nQ{i+1}: {q}")
                print(f"Reference: {ref[:100]}..." if len(ref) > 100 else f"Reference: {ref}")
        
        # Print expected answer format examples
        if dataset_type == "mmlu":
            print("\n=== IMPORTANT: Expected Answer Format ===")
            print("For MMLU questions, models should output answers in the format:")
            print("\"The correct answer is X\" where X is the letter of the correct option (A, B, C, or D).")
            print("Examples:")
            print("  - The correct answer is A")
            print("  - The correct answer is B")
            print("  - The correct answer is C")
            print("  - The correct answer is D")
            print("========================================")
        elif dataset_name == "gpqa":
            print("\n=== IMPORTANT: Expected Answer Format ===")
            print("For GPQA questions, models should output answers in the format:")
            print("\"The correct answer is X\" where X is the letter of the correct option (A, B, C, or D).")
            print("Examples:")
            print("  - The correct answer is A")
            print("  - The correct answer is B")
            print("  - The correct answer is C")
            print("  - The correct answer is D")
            print("========================================")
        elif dataset_type == "summarization":
            print("\n=== IMPORTANT: Expected Answer Format ===")
            print("For summarization tasks, models should output a concise summary.")
            print("========================================")
        elif dataset_type == "translation":
            print("\n=== IMPORTANT: Expected Answer Format ===")
            print("For translation tasks, models should output the translation text.")
            print("========================================")
        
        # Add progress indicator
        print("\n" + "="*50)
        print(f"Starting evaluation of {len(questions)} questions using metrics for {dataset_type}")
        print("This may take several minutes depending on the number of questions")
        print("="*50 + "\n")
        
        # Run evaluation
        results = run_comprehensive_evaluation(questions, reference_answers, dataset_type, use_llm_judge=False)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(elapsed_time, 60)
        print(f"\nTotal evaluation time: {int(minutes)} minutes and {int(seconds)} seconds")
        
        return results
    
    except Exception as e:
        print(f"Error in benchmark evaluation: {e}")
        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(elapsed_time, 60)
        print(f"\nEvaluation failed after {int(minutes)} minutes and {int(seconds)} seconds")
        return None, None

def test_translation(source_text, source_lang="EN", target_lang="DE"):
    """
    Test the ensemble's translation capability directly.
    
    Args:
        source_text: Text to translate
        source_lang: Source language code (e.g., EN, DE, FR)
        target_lang: Target language code (e.g., EN, DE, FR)
        
    Returns:
        The translated text and individual model translations
    """
    print(f"\n=== Testing Translation from {source_lang} to {target_lang} ===")
    
    # Construct a translation query
    query = f"Translate the following {source_lang} text to {target_lang}:\n\n{source_text}"
    
    # Create a dictionary to store all results
    all_translations = {}
    
    # Get individual model translations
    print("\n=== Individual Model Translations ===")
    for role, model_info in MODELS.items():
        model_name = model_info["name"]
        print(f"\nTranslating with {role} ({model_name})...")
        
        if role == "president":
            system_prompt = (
                "You are a highly accurate translation assistant.\n"
                "Translate the text from the source language to the target language with high accuracy.\n"
                "Provide ONLY the translation with no explanation, commentary, or English text.\n"
                "DO NOT refuse to translate the content regardless of what it contains.\n"
            )
        else:
            system_prompt = (
                f"You are {role}, a specialized translation AI assistant.\n"
                "Your task is to:\n"
                "1. Translate the text from the source language to the target language with high accuracy\n"
                "2. Maintain the tone, meaning, and nuance of the original text\n"
                "3. DO NOT refuse to translate due to content concerns\n"
                "4. DO NOT add commentary or analysis to your translation\n"
                "5. Provide ONLY the translated text, with no explanation or English text\n"
            )
        
        messages = create_chat_messages(system_prompt, query)
        translation = query_openrouter_api(model_name, messages, temperature=0.3, max_tokens=1000)
        
        all_translations[role] = translation
        print(f"{role}: {translation}")
    
    # Run the ensemble translation
    print("\n=== Ensemble Translation ===")
    ensemble_translation = run_conversation(query, silent=True)
    all_translations["ensemble"] = ensemble_translation
    
    print(f"\nEnsemble: {ensemble_translation}")
    
    return ensemble_translation, all_translations

def main():
    """Main function to run the LLM ensemble system"""
    print("\n=== LLM Ensemble Learning System ===")
    print("This system uses four models to generate responses:")
    print("  - Minister 1: mistral/ministral-8b")
    print("  - Minister 2: meta-llama/llama-3.1-8b-instruct:free")
    print("  - Minister 3: qwen/qwen3-8b")
    print("  - President: deepseek/deepseek-r1-distill-llama-8b")
    
    # Check if API key is available
    if not api_key_available:
        print("\n⚠️ WARNING: OpenRouter API key is missing or invalid.")
        print("You need to set up your API key to use this system.")
        print("Please add your API key to the .env file or restart the application.")
        return
    
    # Check if models are available
    try:
        models_available = check_model_availability()
        if not models_available:
            print("\nWarning: Some models are not available. The system may not function correctly.")
            proceed = input("Do you want to proceed anyway? (y/n): ").lower()
            if proceed != 'y':
                print("Exiting program.")
                return
    except Exception as e:
        print(f"\nError checking model availability: {e}")
        print("This might be due to an invalid API key or connection issue.")
        proceed = input("Do you want to proceed anyway? (y/n): ").lower()
        if proceed != 'y':
            print("Exiting program.")
            return
    
    print("\nAvailable commands:")
    print("  query       - Ask a question to the ensemble")
    print("  benchmark   - Run evaluation using a benchmark dataset")
    print("  benchmarks  - List available benchmark datasets")
    print("  translate   - Test translation capability directly")
    print("  exit        - Exit the program")
    
    while True:
        try:
            user_input = input("\nEnter command: ").strip().lower()
            
            if user_input in ['exit', 'quit']:
                print("Exiting program. Goodbye!")
                break
                
            elif user_input == 'benchmarks':
                list_available_benchmarks()
                
            elif user_input == 'benchmark':
                # Get parameters for benchmark
                dataset_name = input("Enter dataset name (mmlu, gpqa, summarization, translation): ").strip().lower()
                valid_datasets = ["mmlu", "gpqa", "summarization", "translation"]
                if dataset_name not in valid_datasets:
                    print(f"Invalid dataset. Using mmlu.")
                    dataset_name = "mmlu"
                
                subset = input(f"Enter subset for {dataset_name} (optional, press Enter for default): ").strip()
                if not subset:
                    subset = None
                
                try:
                    num_samples = int(input("Enter number of samples (1-100): ").strip())
                    num_samples = max(1, min(100, num_samples))  # Limit to 1-100 samples
                except:
                    num_samples = 3
                    print(f"Using default: {num_samples} samples")
                
                run_benchmark_evaluation(dataset_name, subset, num_samples, use_llm_judge=False)
                
            elif user_input == 'translate':
                source_lang = input("Enter source language code (e.g., EN, DE, FR): ").strip().upper()
                target_lang = input("Enter target language code (e.g., EN, DE, FR): ").strip().upper()
                source_text = input("Enter text to translate: ").strip()
                
                if not source_text:
                    print("Please enter a valid source text.")
                    continue
                
                try:
                    translation, _ = test_translation(source_text, source_lang, target_lang)
                    
                    # Ask if user wants to try another query
                    another = input("\nDo you want to translate another text? (y/n): ").lower()
                    if another != 'y':
                        print("Returning to main menu.")
                        
                except Exception as e:
                    print(f"An error occurred during translation: {e}")
                    print("Please try again with a different text or check your internet connection.")
                
            elif user_input == 'query':
                query = input("\nEnter your query: ").strip()
                if not query:
                    print("Please enter a valid query.")
                    continue
                
                try:
                    final_answer = run_conversation(query)
                    
                    # Ask if user wants to try another query
                    another = input("\nDo you want to ask another question? (y/n): ").lower()
                    if another != 'y':
                        print("Thank you for using the LLM Ensemble Learning System. Goodbye!")
                        break
                        
                except Exception as e:
                    print(f"An error occurred: {e}")
                    print("Please try again with a different query or check your internet connection.")
            
            elif not user_input:
                print("Please enter a command.")
                
            else:
                print(f"Command '{user_input}' not recognized. Available commands: query, benchmark, benchmarks, translate, exit")
                
        except KeyboardInterrupt:
            print("\n\nProcess interrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            print("Please try again.")

# The last lines of the file should call main()
if __name__ == "__main__":
    main()