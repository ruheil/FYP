from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteriaList
import torch
import json
import re
import os
from huggingface_hub import login, snapshot_download

# Load API token from environment variable instead of hardcoding it
# You can set this by running:
# For Windows: set HUGGINGFACE_TOKEN=your_token_here
# For Linux/Mac: export HUGGINGFACE_TOKEN=your_token_here

# Load from .env file if it exists
try:
    from dotenv import load_dotenv
    # Create a .env file with: HUGGINGFACE_TOKEN=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
    # Make sure to add .env to .gitignore to prevent accidentally pushing your token
    
    # Try multiple encoding options in case the file was saved with a non-UTF-8 encoding
    try:
        load_dotenv(encoding="utf-8")
        print("Loaded environment variables from .env file with UTF-8 encoding")
    except UnicodeDecodeError:
        # Try with UTF-16 encoding if UTF-8 fails
        try:
            load_dotenv(encoding="utf-16")
            print("Loaded environment variables from .env file with UTF-16 encoding")
        except Exception as e:
            print(f"Warning: Failed to load .env file: {e}")
except ImportError:
    print("python-dotenv not installed. To use .env file, install with: pip install python-dotenv")

huggingface_token = os.environ.get("HUGGINGFACE_TOKEN")
if not huggingface_token:
    print("WARNING: No Hugging Face token found in environment variables.")
    print("Please set the HUGGINGFACE_TOKEN environment variable.")
    print("For Windows: set HUGGINGFACE_TOKEN=your_token_here")
    print("For Linux/Mac: export HUGGINGFACE_TOKEN=your_token_here")
    
    # Provide a fallback to manually enter the token
    print("\nWould you like to enter your Hugging Face token now? (y/n)")
    response = input().strip().lower()
    if response == 'y':
        print("Enter your Hugging Face token:")
        huggingface_token = input().strip()
        # Don't save to .env file for security, just use it for this session
    else:
        print("No token provided. Exiting.")
        exit(1)

# Log in to Hugging Face
login(huggingface_token)

from huggingface_hub import snapshot_download

model_repo = "teknium/OpenHermes-2.5-Mistral-7B" 
snapshot_download(
    repo_id=model_repo,
    allow_patterns=["*.json", "*.model", "*.py", "*.safetensors", "tokenizer.model"],
    local_dir=model_repo
)

# For each Mistral-based model directory
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Custom stopping criteria
class StopOnTokens:
    def __init__(self, token_ids):
        self.token_ids = token_ids

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0][-1] in self.token_ids

# Model loading function
def load_model(model_name):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Special handling for Mistral-based models
    if "mistral" in model_name.lower() or "zephyr" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True  # Critical for SentencePiece compatibility
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return model, tokenizer

# Model unloading function
def unload_model(model):
    model = model.to("cpu")
    del model
    torch.cuda.empty_cache()

#  feedback generation function
def get_president_feedback(responses):
    president_model, president_tokenizer = load_model("mistralai/Mistral-7B-Instruct-v0.1")
    
    #  prompt for more critical analysis
    critique_prompt = [
        {"role": "system", "content": """You are a critical analysis expert. Evaluate responses with extreme attention to detail.
STRICT OUTPUT RULES:
1. Respond ONLY with valid JSON
2. Use format: {'score': int 1-5, 'issues': [strings]}
3. Scoring guide:
   - 5: Perfect, comprehensive, and innovative
   - 4: Very good but room for minor improvements
   - 3: Good but needs significant improvements
   - 2: Major issues present
   - 1: Serious problems or incomplete
4. ALWAYS include at least 2 specific improvement points in 'issues', even for good responses"""},
        {"role": "user", "content": f"""Critically analyze these responses. Evaluate:
1. Depth of analysis
2. Factual accuracy
3. Logical coherence
4. Comprehensiveness
5. Clarity and structure
6. Supporting evidence
7. Counter-arguments consideration

Responses: {json.dumps(responses)}

JSON output:"""}
    ]
    
    # Use Mistral's template with special tokens
    formatted_prompt = president_tokenizer.apply_chat_template(
        critique_prompt,
        tokenize=False,
        add_generation_prompt=True
    ) + "\n{"

    inputs = president_tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    
    # Constrained generation settings
    outputs = president_model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.3,
        repetition_penalty=1.5,
        num_beams=2,
        stopping_criteria=StoppingCriteriaList([
            StopOnTokens(token_ids=[president_tokenizer.eos_token_id])
        ])
    )
    
    feedback = president_tokenizer.decode(outputs[0], skip_special_tokens=True)
    unload_model(president_model)

    # Multi-stage parsing with validation
    parsed = extract_json(feedback)
    if parsed is not None:
        return parsed
    else:
        print("Primary parsing failed, initiating fallback...")
        return self_validate_feedback(feedback)

# JSON extraction and validation
def extract_json(feedback):
    # Normalization steps
    feedback = feedback.replace('\n', ' ').replace('\\"', '"')
    
    # Progressive parsing attempts
    attempts = [
        lambda: json.loads(feedback),                        # Direct parse
        lambda: json.loads(feedback.split('[/INST]')[-1]),   # Remove instruction tags
        lambda: json.loads(re.search(r'{.*}', feedback).group()), # Regex match
        lambda: json.loads(feedback[feedback.find('{'):feedback.rfind('}')+1]) # Positional
    ]
    
    for attempt in attempts:
        try:
            result = attempt()
            if validate_json(result):
                return result
        except:
            continue
    return None

def validate_json(data):
    if not isinstance(data, dict):
        return False
    
    score = data.get('score')
    issues = data.get('issues')
    
    # Enhanced validation
    valid_score = isinstance(score, int) and 1 <= score <= 5
    valid_issues = isinstance(issues, list) and len(issues) >= 1 and all(isinstance(i, str) and len(i.strip()) > 0 for i in issues)
    
    return valid_score and valid_issues

# Fallback validation function
def self_validate_feedback(feedback):
    # More specific validation prompt
    validation_prompt = f"""Fix this JSON to match the exact format:
{{'score': (integer 1-5), 'issues': [at least 2 specific improvement points]}}

Current invalid input: {feedback}
Valid output (only JSON):"""
    
    # Use smaller model for validation
    validator_model, validator_tokenizer = load_model("NousResearch/Hermes-2-Pro-Mistral-7B")
    inputs = validator_tokenizer(validation_prompt, return_tensors="pt").to("cuda")
    outputs = validator_model.generate(**inputs, max_new_tokens=150)
    parsed_feedback = validator_tokenizer.decode(outputs[0], skip_special_tokens=True)
    unload_model(validator_model)
    
    return extract_json(parsed_feedback) or {"score": 3, "issues": ["Feedback parsing failed"]}

# Main response generation function
def get_minister_response(query, model_name):
    model, tokenizer = load_model(model_name)
    prompt = f"<|system|>Analyze this query step-by-step:</s><|user|>{query}</s><|assistant|>"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=512)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    unload_model(model)
    return response

# Main execution loop
ministers = [
    "HuggingFaceH4/zephyr-7b-beta",
    "teknium/OpenHermes-2.5-Mistral-7B"
]

user_query = "Compare and contrast different approaches to renewable energy storage, discussing their efficiency, cost, and environmental impact"
max_cycles = 3
min_acceptable_score = 4  # Minimum score to accept as final answer

for cycle in range(max_cycles):
    print(f"\n=== Iteration {cycle+1} ===")
    responses = [get_minister_response(user_query, model) for model in ministers]
    
    for i, response in enumerate(responses):
        print(f"\nMinister {i+1} Response:")
        print(response)
    
    feedback = get_president_feedback(responses)
    print(f"\nFeedback Score: {feedback['score']}")
    print(f"Issues: {feedback['issues']}")
    
    # Only break if we're at max cycles or got a perfect score
    if cycle == max_cycles - 1 or feedback["score"] == 5:
        print("\nFinal Answer:", responses[0].replace("<|system|>", "").replace("<|user|>", "").replace("<|assistant|>", ""))
        break
    else:
        # More specific improvement request
        user_query = f"""Original query: {user_query}

    Previous response had these issues to address:
    {chr(10).join('- ' + issue for issue in feedback['issues'])}

    Provide an improved, more comprehensive answer that specifically addresses these issues."""