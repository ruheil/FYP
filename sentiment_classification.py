import os
import torch
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import re

# Load environment variables from .env
try:
    from dotenv import load_dotenv
    try:
        load_dotenv(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            load_dotenv(encoding="utf-16")
        except Exception as e:
            print(f"Warning: Failed to load .env file: {e}")
except ImportError:
    print("python-dotenv not installed. To use .env file, install with: pip install python-dotenv")

# Get the Hugging Face token from environment variables
huggingface_token = os.environ.get("HUGGINGFACE_TOKEN")
if not huggingface_token:
    print("WARNING: No Hugging Face token found in environment variables.")
    print("Please enter your Hugging Face token:")
    huggingface_token = input().strip()

# Model loading function (reused from main.py)
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
            use_fast=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return model, tokenizer

# Model unloading function (reused from main.py)
def unload_model(model):
    model = model.to("cpu")
    del model
    torch.cuda.empty_cache()

# Load SST-2 dataset (Stanford Sentiment Treebank)
def load_sst2_dataset(split_size=100):
    """
    Load the SST-2 dataset for sentiment analysis
    split_size: Number of examples to use (for quick testing)
    """
    dataset = load_dataset("sst2")
    
    # Create a smaller test set for quick evaluation
    test_data = dataset["validation"].select(range(split_size))
    
    return test_data

# Single model classification
def single_model_classification(model_name, test_data, debug_mode=True):
    """
    Perform sentiment classification using a single model
    """
    print(f"\nPerforming sentiment classification with single model: {model_name}")
    
    model, tokenizer = load_model(model_name)
    
    predictions = []
    raw_responses = []
    
    for example in tqdm(test_data):
        text = example["sentence"]
        # Improved prompt
        prompt = f"<|system|>You are an expert sentiment analysis assistant specialized in binary classification. Analyze the following text and determine whether it expresses a POSITIVE or NEGATIVE sentiment. Your response must ONLY contain the word 'POSITIVE' or 'NEGATIVE'.</s><|user|>{text}</s><|assistant|>"
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7, do_sample=True)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        raw_responses.append(response)
        
        # More robust extraction of the prediction
        response_lower = response.lower()
        
        # Count occurrences of positive/negative words to handle ambiguous responses
        positive_count = response_lower.count("positive")
        negative_count = response_lower.count("negative")
        
        # If the model is more confident in one class, use that
        if positive_count > negative_count:
            predictions.append(1)  # Positive sentiment
        elif negative_count > positive_count:
            predictions.append(0)  # Negative sentiment
        else:
            # If equal counts or no explicit mention, check which one appears first
            pos_index = response_lower.find("positive")
            neg_index = response_lower.find("negative")
            
            if pos_index != -1 and (neg_index == -1 or pos_index < neg_index):
                predictions.append(1)
            else:
                predictions.append(0)
    
    unload_model(model)
    
    # Print debug information for the first few examples
    if debug_mode:
        print("\nDEBUG - Single Model Responses:")
        for i in range(min(5, len(raw_responses))):
            print(f"Example {i+1}:")
            print(f"Text: {test_data[i]['sentence']}")
            print(f"True label: {test_data[i]['label']}")
            print(f"Model response: {raw_responses[i][:200]}...")
            print(f"Predicted label: {predictions[i]}")
            print("-" * 50)
    
    # Calculate metrics
    y_true = test_data["label"]
    y_pred = predictions
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Calculate class distribution
    class_distribution = {
        "total": len(predictions),
        "positive_predictions": sum(predictions),
        "negative_predictions": len(predictions) - sum(predictions),
        "positive_actual": sum(y_true),
        "negative_actual": len(y_true) - sum(y_true)
    }
    
    results = {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "predictions": predictions,
        "raw_responses": raw_responses,
        "class_distribution": class_distribution
    }
    
    return results

# Ensemble model classification
def ensemble_classification(model_names, president_model_name, test_data, debug_mode=True):
    """
    Perform sentiment classification using ensemble learning with multiple models
    """
    print(f"\nPerforming sentiment classification with ensemble learning:")
    print(f"Minister models: {', '.join(model_names)}")
    print(f"President model: {president_model_name}")
    
    predictions = []
    all_minister_responses = []
    all_president_responses = []
    
    for example in tqdm(test_data):
        text = example["sentence"]
        
        # Get predictions from minister models
        minister_responses = []
        
        for model_name in model_names:
            model, tokenizer = load_model(model_name)
            # Improved minister prompt
            prompt = f"<|system|>You are a sentiment analysis expert. Carefully analyze the sentiment of the following text. First, provide a detailed explanation of the sentiment expressed in the text, including specific words or phrases that indicate sentiment. Then, provide your final classification: POSITIVE or NEGATIVE.</s><|user|>{text}</s><|assistant|>"
            
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            minister_responses.append(response)
            
            unload_model(model)
        
        all_minister_responses.append(minister_responses)
        
        # Use president model to decide final classification
        president_model, president_tokenizer = load_model(president_model_name)
        
        # Improved president prompt
        president_prompt = f"<|system|>You are a classification expert. Your task is to determine the sentiment of a text based on two analyses from different models. Read both analyses carefully, then make your own determination of whether the sentiment is POSITIVE or NEGATIVE. After your analysis, your final answer must ONLY contain the word 'POSITIVE' or 'NEGATIVE'.</s><|user|>Text: {text}\n\nModel 1 analysis: {minister_responses[0]}\n\nModel 2 analysis: {minister_responses[1]}\n\nBased on these analyses, what is the sentiment? Answer with POSITIVE or NEGATIVE only.</s><|assistant|>"
        
        inputs = president_tokenizer(president_prompt, return_tensors="pt").to("cuda")
        outputs = president_model.generate(**inputs, max_new_tokens=100, temperature=0.7, do_sample=True)
        president_response = president_tokenizer.decode(outputs[0], skip_special_tokens=True)
        all_president_responses.append(president_response)
        
        unload_model(president_model)
        
        # More robust extraction of the prediction
        response_lower = president_response.lower()
        
        # Count occurrences of positive/negative words
        positive_count = response_lower.count("positive")
        negative_count = response_lower.count("negative")
        
        # If the model is more confident in one class, use that
        if positive_count > negative_count:
            predictions.append(1)  # Positive sentiment
        elif negative_count > positive_count:
            predictions.append(0)  # Negative sentiment
        else:
            # If equal counts or no explicit mention, check which one appears first
            pos_index = response_lower.find("positive")
            neg_index = response_lower.find("negative")
            
            if pos_index != -1 and (neg_index == -1 or pos_index < neg_index):
                predictions.append(1)
            else:
                predictions.append(0)
    
    # Print debug information for the first few examples
    if debug_mode:
        print("\nDEBUG - Ensemble Model Responses:")
        for i in range(min(5, len(all_president_responses))):
            print(f"Example {i+1}:")
            print(f"Text: {test_data[i]['sentence']}")
            print(f"True label: {test_data[i]['label']}")
            print(f"Minister 1 response: {all_minister_responses[i][0][:100]}...")
            print(f"Minister 2 response: {all_minister_responses[i][1][:100]}...")
            print(f"President response: {all_president_responses[i][:200]}...")
            print(f"Predicted label: {predictions[i]}")
            print("-" * 50)
    
    # Calculate metrics
    y_true = test_data["label"]
    y_pred = predictions
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Calculate class distribution
    class_distribution = {
        "total": len(predictions),
        "positive_predictions": sum(predictions),
        "negative_predictions": len(predictions) - sum(predictions),
        "positive_actual": sum(y_true),
        "negative_actual": len(y_true) - sum(y_true)
    }
    
    results = {
        "model": "Ensemble",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "predictions": predictions,
        "president_responses": all_president_responses,
        "minister_responses": all_minister_responses,
        "class_distribution": class_distribution
    }
    
    return results

# Visualize the comparison
def visualize_comparison(single_results, ensemble_results):
    """
    Create visualizations comparing single model vs ensemble performance
    """
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    single_scores = [single_results[metric] for metric in metrics]
    ensemble_scores = [ensemble_results[metric] for metric in metrics]
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, single_scores, width, label=f'Single Model ({single_results["model"]})')
    plt.bar(x + width/2, ensemble_scores, width, label='Ensemble Model')
    
    plt.xlabel('Metrics')
    plt.ylabel('Scores')
    plt.title('Performance Comparison: Single Model vs Ensemble')
    plt.xticks(x, metrics)
    plt.ylim(0, 1.0)
    plt.legend()
    
    plt.savefig('performance_comparison.png')
    plt.show()
    
    # Create a confusion matrix for disagreements
    single_pred = single_results["predictions"]
    ensemble_pred = ensemble_results["predictions"]
    true_labels = test_data["label"]
    
    # Find indices where the models disagree
    disagreement_indices = [i for i in range(len(single_pred)) if single_pred[i] != ensemble_pred[i]]
    
    if disagreement_indices:
        print(f"\nFound {len(disagreement_indices)} disagreements between models")
        print("\nSample disagreements:")
        
        # Show up to 5 disagreement examples
        for i in disagreement_indices[:5]:
            print(f"Text: {test_data[i]['sentence']}")
            print(f"True label: {'POSITIVE' if true_labels[i] == 1 else 'NEGATIVE'}")
            print(f"Single model: {'POSITIVE' if single_pred[i] == 1 else 'NEGATIVE'}")
            print(f"Ensemble: {'POSITIVE' if ensemble_pred[i] == 1 else 'NEGATIVE'}")
            print(f"Correct model: {'Single' if single_pred[i] == true_labels[i] else 'Ensemble' if ensemble_pred[i] == true_labels[i] else 'None'}")
            print("-" * 50)

if __name__ == "__main__":
    # Load the dataset
    print("Loading SST-2 dataset...")
    # Use an even smaller sample size for quicker testing
    test_data = load_sst2_dataset(split_size=20)  # Reduced from 50 to 20 for faster testing
    
    # Ensure we have a balanced dataset
    labels = [example["label"] for example in test_data]
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    print(f"Dataset composition: {positive_count} positive, {negative_count} negative examples")
    
    # Define models
    minister1_model = "HuggingFaceH4/zephyr-7b-beta"
    minister2_model = "teknium/OpenHermes-2.5-Mistral-7B"
    president_model = "mistralai/Mistral-7B-Instruct-v0.1"
    
    # Enable debug mode
    debug_mode = True
    
    # Single model classification (using just one of the minister models)
    single_results = single_model_classification(minister1_model, test_data, debug_mode)
    
    # Ensemble model classification
    ensemble_results = ensemble_classification(
        [minister1_model, minister2_model],
        president_model,
        test_data,
        debug_mode
    )
    
    # Display results
    print("\n====== RESULTS ======")
    print("\nSingle Model Performance:")
    print(f"Model: {single_results['model']}")
    print(f"Accuracy: {single_results['accuracy']:.4f}")
    print(f"Precision: {single_results['precision']:.4f}")
    print(f"Recall: {single_results['recall']:.4f}")
    print(f"F1 Score: {single_results['f1_score']:.4f}")
    print(f"Class distribution: {single_results['class_distribution']}")
    
    print("\nEnsemble Model Performance:")
    print(f"Models: {minister1_model}, {minister2_model} with {president_model} as president")
    print(f"Accuracy: {ensemble_results['accuracy']:.4f}")
    print(f"Precision: {ensemble_results['precision']:.4f}")
    print(f"Recall: {ensemble_results['recall']:.4f}")
    print(f"F1 Score: {ensemble_results['f1_score']:.4f}")
    print(f"Class distribution: {ensemble_results['class_distribution']}")
    
    # Improvement percentages
    if single_results['accuracy'] > 0:
        acc_improvement = (ensemble_results['accuracy'] - single_results['accuracy']) / single_results['accuracy'] * 100
    else:
        acc_improvement = 0
        
    if single_results['f1_score'] > 0:
        f1_improvement = (ensemble_results['f1_score'] - single_results['f1_score']) / single_results['f1_score'] * 100
    else:
        f1_improvement = 0
    
    print(f"\nAccuracy improvement: {acc_improvement:.2f}%")
    print(f"F1 score improvement: {f1_improvement:.2f}%")
    
    # Visualize the comparison
    visualize_comparison(single_results, ensemble_results) 