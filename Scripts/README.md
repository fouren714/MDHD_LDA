# LDA Processing Scripts

This repository contains scripts for preprocessing text data, generating topic models using Latent Dirichlet Allocation (LDA), and calculating various metrics related to the topics.

## Files

### 1. LDA2.py
This script implements a class for performing LDA and related tasks.

#### Key Features:
- **Coherence Calculation**: Calculates topic coherence scores.
- **Saliency Calculation**: Computes saliency scores for topics.
- **Model Comparison**: Compares scores between two LDA models.
- **Preprocessing**: Includes methods for preprocessing text data.

### 2. make_weights2.py
This script generates weights for LDA topics.

#### Key Features:
- **Data Loading**: Loads necessary data for processing.
- **Dictionary and Corpus Building**: Constructs dictionaries and corpora from the text data.
- **Weight Calculation**: Calculates weights for words in the topics.

### 3. preprocess2.py
This script handles the preprocessing of text data.

#### Key Features:
- **Lemmatization**: Lemmatizes the text data.
- **Stop Words Removal**: Removes common stop words.
- **Special Characters Handling**: Deals with special characters in the text.
- **Corpus Building**: Builds a corpus from text files.

## Getting Started

### Prerequisites
- Python 3.x
- Required Python libraries: `pandas`, `numpy`, `gensim`, `nltk`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/fouren714/MDHD_LDA.git
