# IMDB Sentiment Analysis with PyTorch RNN

This project implements a Simple RNN for sentiment analysis on the IMDB movie reviews dataset using PyTorch, while still utilizing TensorFlow's IMDB dataset for data loading and preprocessing.

## Project Structure

- simplernn.ipynb - Main training notebook with PyTorch RNN implementation
- prediction.ipynb - Notebook for making predictions with the trained model
- embedding.ipynb - Demonstrates word embeddings using both TensorFlow preprocessing and PyTorch embeddings
- main.py - Streamlit web application for interactive sentiment analysis
- requirements.txt - Python dependencies

## Key Changes from TensorFlow to PyTorch

### 1. Model Architecture
- *Before (TensorFlow)*: Used keras.Sequential with Embedding, SimpleRNN, and Dense layers
- *After (PyTorch)*: Custom SimpleRNNClassifier class inheriting from nn.Module

### 2. Data Handling
- *Dataset*: Still uses TensorFlow's imdb.load_data() for consistency
- *Preprocessing*: Uses TensorFlow's sequence.pad_sequences() 
- *Tensors*: Converts to PyTorch LongTensor and FloatTensor
- *DataLoaders*: Uses PyTorch's DataLoader for efficient batching

### 3. Training Loop
- *Before*: Used model.fit() with callbacks
- *After*: Manual training loop with proper gradient handling
- *Early Stopping*: Implemented manually with patience mechanism
- *Device Support*: Automatic GPU/CPU detection

### 4. Model Saving/Loading
- *Before*: Saved as .h5 file using model.save()
- *After*: Saves state dictionary as .pth file using torch.save()

## Installation

bash
pip install -r requirements.txt


## Quick Start

### Option 1: Using the Training Script (Recommended)
bash
# Install dependencies
pip install -r requirements.txt

# Train the model with default parameters
python train_model.py

# Or with custom parameters
python train_model.py --epochs 15 --batch_size 64 --lr 0.0005


### Option 2: Using Jupyter Notebooks
1. Install dependencies: pip install -r requirements.txt
2. Run simplernn.ipynb to train the model
3. Run prediction.ipynb to test predictions
4. Run embedding.ipynb to understand embeddings

### Option 3: Web Application
bash
# Make sure the model is trained first
streamlit run main.py


### Bonus: See the Comparison
bash
python comparison.py


## Usage

### 1. Training the Model
Run the simplernn.ipynb notebook to train the PyTorch RNN model:
- Loads IMDB dataset using TensorFlow
- Preprocesses data and creates PyTorch DataLoaders
- Trains the model with early stopping
- Saves the trained model as simple_rnn_imdb_pytorch.pth

### 2. Making Predictions
Use the prediction.ipynb notebook to test the trained model:
- Loads the trained PyTorch model
- Provides helper functions for text preprocessing
- Demonstrates prediction on sample reviews

### 3. Web Application
Run the Streamlit app for interactive predictions:
bash
streamlit run main.py


### 4. Understanding Embeddings
The embedding.ipynb notebook shows:
- How word embeddings work
- Comparison between TensorFlow preprocessing and PyTorch embeddings
- Visualization of embedding layers

## Model Architecture

python
class SimpleRNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SimpleRNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()


### Parameters:
- *Vocabulary Size*: 10,000 words
- *Embedding Dimension*: 128
- *Hidden Dimension*: 128
- *Sequence Length*: 500 tokens
- *Output*: Binary classification (positive/negative)

## Features

### Training Features:
- ✅ Early stopping with patience
- ✅ Validation split (20%)
- ✅ GPU/CPU automatic detection
- ✅ Progress bars with tqdm
- ✅ Training history visualization
- ✅ Model checkpointing

### Prediction Features:
- ✅ Text preprocessing pipeline
- ✅ Confidence scoring
- ✅ Batch prediction support
- ✅ Interactive web interface

### Data Pipeline:
- ✅ TensorFlow IMDB dataset integration
- ✅ PyTorch tensor conversion
- ✅ Efficient DataLoader usage
- ✅ Proper train/validation/test splits

## Performance

The model achieves similar performance to the original TensorFlow implementation:
- *Training Accuracy*: ~85-90%
- *Validation Accuracy*: ~80-85%
- *Training Time*: ~10-15 minutes (depending on hardware)

## Benefits of PyTorch Implementation

1. *More Control*: Manual training loop allows for better debugging and customization
2. *Flexibility*: Easier to modify model architecture and training process
3. *Research-Friendly*: Better suited for experimentation and research
4. *Dynamic Graphs*: More intuitive debugging with dynamic computational graphs
5. *Modern Ecosystem*: Access to latest PyTorch features and community tools

## Hybrid Approach

This implementation demonstrates a hybrid approach:
- *Data*: Uses TensorFlow's well-established IMDB dataset
- *Preprocessing*: Leverages TensorFlow's robust preprocessing utilities
- *Model & Training*: Uses PyTorch for modern deep learning practices
- *Deployment*: Streamlit for easy web deployment

This approach combines the best of both frameworks while maintaining compatibility with existing TensorFlow datasets.

## Output

<img width="1055" height="707" alt="image" src="https://github.com/user-attachments/assets/bd221d5e-7b7a-4411-875a-73a8959a7815" />

