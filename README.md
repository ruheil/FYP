# LLM Ensemble Learning for Enhanced Classification

This repository contains a proof-of-concept implementation demonstrating how ensemble learning with multiple Large Language Models (LLMs) is more robust than using a single model for classification tasks.

## Project Overview

The main goal of this project is to prove that classification via ensemble learning with LLMs provides more accurate and robust results compared to using a single LLM. The project includes:

1. Single model classification using one LLM
2. Ensemble classification combining multiple models with a "minister-president" architecture
3. Comparative evaluation of both approaches

## Architecture

The system uses a hierarchical decision-making structure:
- **Minister Models**: Two separate LLMs that analyze input text and provide their individual assessments
- **President Model**: A third LLM that reviews both ministers' analyses and makes the final classification decision

## Key Features

- Implementation of both single-model and ensemble-based classification
- Comprehensive evaluation and comparison of performance metrics
- Robust sentiment extraction from model outputs
- Visualization tools for comparing performance
- Analysis of disagreement cases between models

## Setup and Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)

### Installation

1. Clone this repository:
```
git clone https://github.com/ruheil/FYP.git
cd FYP
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Set up your Hugging Face token:
   - Create a `.env` file in the root directory
   - Add your token: `HUGGINGFACE_TOKEN=your_token_here`

## Usage

### Running Sentiment Analysis Comparison

```
python sentiment_classification.py
```

This will:
1. Load a portion of the SST-2 dataset
2. Run both single-model and ensemble classification
3. Display comparative performance metrics
4. Generate a visualization comparing the results

### Customization

You can modify the following parameters in `sentiment_classification.py`:
- Test dataset size (`split_size` parameter)
- Models used for classification
- Debug mode settings

## Results

The ensemble approach demonstrates better performance across multiple metrics:
- Higher accuracy and F1 score
- Better handling of ambiguous cases
- More consistent performance across different types of text

## Project Structure

- `main.py`: Core implementation of the minister-president model
- `sentiment_classification.py`: Sentiment analysis implementation comparing single vs ensemble approaches
- `requirements.txt`: Required Python packages
- `.env`: Environment variables (not tracked by git)

## Future Work

- Extend to other classification tasks beyond sentiment analysis
- Implement more complex ensemble architectures
- Optimize for performance and resource usage
- Add support for additional models and datasets

## License

[MIT License](LICENSE) 