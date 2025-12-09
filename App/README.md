# FedBN Activity Recognition - XAI Dashboard

This project implements an explainable AI dashboard for a Federated Learning model with BatchNormalization (FedBN) for human activity recognition using the PAMAP2 dataset.

## Features

- **Interactive Model Exploration:** Visualize and understand model predictions
- **Multiple XAI Methods:** GradCAM and Integrated Gradients for model explanations
- **Client-specific Analysis:** Analyze model performance on individual clients
- **Round Comparison:** Compare model behavior across different training rounds
- **Interactive Visualizations:** Time-series plots with feature attributions

## Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- Streamlit

### Installation

1. Install the required packages:

```bash
pip install -r app_requirements.txt
```

2. Run the Streamlit app:

```bash
streamlit run app.py
```

### Using the Dashboard

The dashboard provides three main tabs:

1. **Single Sample Analysis:** Upload or select a sample to analyze with XAI methods
2. **Client Data Analysis:** Explore model behavior on specific client data
3. **Round Comparison:** Compare model predictions and explanations across training rounds

## XAI Methods

- **GradCAM:** Visualizes which regions in the time-series input contribute most to the prediction
- **Integrated Gradients:** Provides feature-level attribution for each time point and sensor

## Data Structure

- Model files are stored in `fedbn/models/`
- Client data is stored in `clients_data/`
- Global test data is available in `global_data/`

## Model Architecture

The model uses a CNN architecture for activity recognition:
- Input: 200 timesteps with sensor features
- Conv1D layer 1: 64 filters, kernel=3, ReLU, MaxPool(2)
- Conv1D layer 2: 128 filters, kernel=3, ReLU, MaxPool(2)
- Dense layers: 128 units → 10 classes (activities)
