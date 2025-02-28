# CLAUDE.md - Assistant Guidelines for Stock-AI Project

## Build & Test Commands
- Run specific test: `python -m unittest tests/test_sentiment_analyzer.py`
- Run all tests: `python -m unittest discover tests`
- Run specific test class: `python -m unittest tests.test_sentiment_analyzer.TestFinBERTSentimentAnalyzer`
- Run specific test method: `python -m unittest tests.test_sentiment_analyzer.TestFinBERTSentimentAnalyzer.test_analyzer_initialization`
- Run notebook: `jupyter notebook notebooks/3-model-training.ipynb` or `python notebooks/3-model-training.py`

## Code Style Guidelines
- **Imports**: Group imports by standard library, third-party, and local modules with a blank line between groups
- **Type Hints**: Use type hints for function parameters and return values
- **Documentation**: Use docstrings with Parameters and Returns sections as shown in utils/sentiment_analyzer.py
- **Error Handling**: Employ try/except blocks with specific exceptions
- **Naming Conventions**:
  - Classes: PascalCase (e.g., FinBERTSentimentAnalyzer)
  - Functions/methods: snake_case (e.g., batch_analyze)
  - Variables: snake_case (e.g., sequence_size)
  - Constants: UPPER_SNAKE_CASE
- **Models**: Save trained models to models/ directory
- **Data**: Raw data in data/raw/, processed data in data/processed/