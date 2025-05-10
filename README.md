# LLM Collaborative

A collaborative AI system leveraging multiple Large Language Models (LLMs) to generate higher quality responses than any single model working alone.

## How It Works

This system uses four specialized models working together:

- **Minister 1** (mistral/ministral-8b): An analytical AI assistant that provides comprehensive, factual answers
- **Minister 2** (meta-llama/llama-3.1-8b-instruct): A critical-thinking AI that evaluates information from different angles
- **Minister 3** (qwen/qwen3-8b): A pragmatic AI focused on practical implications and real-world applications
- **President** (deepseek/deepseek-r1-distill-llama-8b): Synthesizes the ministers' perspectives into a definitive answer

The system works through a deliberative process where ministers provide their analysis in sequence, each building on previous responses. The president then synthesizes these perspectives to deliver a final answer that combines the strengths of all models.

## Features

- **Collaborative Intelligence**: Models work together, enhancing each other's strengths and compensating for limitations
- **Multi-turn Deliberation**: Two rounds of discussion between ministers allow for deeper analysis
- **Specialized Tasks**:
  - General question answering
  - Multiple-choice questions (MMLU-style)
  - Text translation
  - Document summarization
- **Benchmark Evaluation**: Compare the collaborative system against individual models using standard datasets

## Requirements

- Python 3.8+
- OpenRouter API key
- Hugging Face API token (optional, for accessing certain datasets)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/llm-collaborative.git
   cd llm-collaborative
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API keys:
   ```
   OPENROUTER_API_KEY=your_openrouter_api_key
   HUGGINGFACE_TOKEN=your_huggingface_token  # Optional
   ```

## Usage

Run the main script:
```
python main.py
```

### Available Commands

- **query**: Ask a question to the collaborative
- **benchmark**: Run evaluation using a benchmark dataset
- **benchmarks**: List available benchmark datasets
- **translate**: Test translation capability directly
- **exit**: Exit the program

## Example Queries

- General knowledge: "What are the main causes of climate change?"
- Multiple choice: "What is the capital of France? A) London B) Paris C) Berlin D) Madrid"
- Translation: "Translate the following English text to German: Hello, how are you?"
- Summarization: "Summarize the following text: [long text here]"

## Benchmark Evaluation

The system can evaluate performance on standard benchmarks:

- **MMLU**: Massive Multitask Language Understanding - tests knowledge across domains
- **GPQA**: Graduate-level Google-Proof Q&A - tests graduate-level knowledge in STEM fields
- **Summarization**: Text summarization using CNN/DailyMail and XSum datasets
- **Translation**: Machine translation using WMT datasets

## License

This project is open source and available under the MIT License. 