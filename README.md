# NLP-Frameshifting: Tracking Policy Framing Shifts in Political Speeches Over Time

This project analyzes the evolution of political discourse in the U.S. Congress by using advanced NLP techniques to track changes in how different political issues are framed over time. Through the utilization of multi-label BERT models for political issue classification and specialized RoBERTa models for scoring emotional intensity and political spectrum positioning, we provide quantitative insights into the shifting nature of political framing from 1945-2017.

## Project Overview

The project implements a novel multi-model classification process to:
- Classify congressional speeches into relevant political issue categories
- Score speeches on both emotional intensity (1-5) and political spectrum (liberal-conservative, 1-5)
- Track and visualize these metrics over time to identify framing shifts
- Analyze party-specific trends and polarization dynamics

## Repository Structure

```
└── rgg1-NLP-Frameshifting/
    ├── README.md
    ├── requirements.txt
    ├── aggregate_attention_analysis.py # Analysis for which parts of speech models pay attention to
    ├── attention_analyzer.py           # Analysis showing attention for specific speeches
    ├── aggregate_attention_analysis    # Output of analysis showing which parts of speech models pay attention to
    ├── attention_analysis              # Output of analysis showing attention for specific speeches
    ├── axis_classifier/                # Model for emotional/political spectrum scoring
    ├── data_exploration/               # Data analysis notebooks and results
    ├── frameshifting_calculations/     # Core analysis calculations and visualization
    ├── issue_classifier/               # Political issue classification model
    ├── issue_classifier_eval/          # Model evaluation tools and results for issue classifier model
    ├── large-training-output/          # Training checkpoints and metrics
    ├── phrase_clusters/                # Keyword mappings and topic phrases
    ├── small_speech_data/              # Sample speech datasets
    └── trained_model_evaluation/       # Model performance analysis for emotional/political spectrum model
```

## Prerequisites

### Required Packages
- Python 3.12+
- PyTorch
- Transformers (Hugging Face)
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tqdm

A complete list of requirements can be found in `requirements.txt`

### Data Requirements
- Stanford Congressional Speech Dataset (43rd-114th Congress)
- Access to GPT-4o-mini (or other model of your choice) API for label (training data) generation (optional)

## Setup and Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/rgg1/NLP-Frameshifting.git
    cd rgg1-NLP-Frameshifting
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv myenv
    source myenv/bin/activate  # On Windows: myenv\Scripts\activate
    ```

3. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Code

1. Train Issue Classifier:
    First, run `issue_classifier/gpt_labeing.ipynb`. This will use gpt-4o-mini to label some the political issues of a subset of speeches to be used as training data for our BERT issue classifier model.
    Then, run `issue_classifier/issue_classifier.ipynb`. This uses that training data to actually train our issue classifier model.

3. Train Axis Classifier: 
    First, run `axis_classifier/gpt_axis_training.ipynb`. This will use gpt-4o-mini to label the emotional/political spectrum scores (1-5) of a subset of speeches to be used as training data for our roBERTa axis classifier model.
    Then, run `axis_classifier/axis_classifier.ipynb`. This uses that training data to actually train our axis classifier model.

4. Generate Analysis:
    Run `frameshifting_calculations/main_calculation.ipynb`. This will produce data (JSON files) and plots showing the actual trends in frameshofting over time by selecting speeches from each congress and evaluating them using our trained models.

## Model Details

### Issue Classifier
- Architecture: BERT-based multi-label classifier
- Performance: F1 Score: 0.88, Accuracy: 0.9635
- Labels: 9 political issue categories including Economy, Healthcare, Education, etc.

### Axis Classifier
- Architecture: RoBERTa-based multi-headed classifier
- Performance: 
  - Emotional Intensity: F1 Score: 0.83, MAE: 0.16
  - Political Spectrum: F1 Score: 0.86, MAE: 0.16
- Scoring: Both axes use 1-5 scale

## Data Structure

The project uses multiple data formats:
- Raw speech data: Text files with congress member speeches. This is gathered from the Stanford Congressional Speech Dataset. We use the `hein-daily` and `hein-bound` folders from the dataset in our model training and evaluation. They are in the `.gitignore` due to size.
- Intermediate JSON files: Processed speeches with labels and scores. 
- Analysis outputs: JSON and CSV files containing temporal trends and statistics.
