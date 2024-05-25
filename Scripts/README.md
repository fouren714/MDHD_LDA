
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
   ```
2. Navigate to the project directory:
   ```bash
   cd your-repository
   ```
3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### LDA2.py
1. Import the necessary class and create an instance:
   ```python
   from LDA2 import LDAProcessor
   lda_processor = LDAProcessor()
   ```
2. Use the methods provided to perform LDA and calculate metrics.

#### make_weights2.py
1. Run the script to generate weights:
   ```bash
   python make_weights2.py
   ```

#### preprocess2.py
1. Import the preprocessing class and create an instance:
   ```python
   from preprocess2 import Preprocessor
   preprocessor = Preprocessor()
   ```
2. Use the methods provided to preprocess your text data.

## Contributing
Feel free to fork this repository and make contributions. Pull requests are welcome.

## License
This project is licensed under the MIT License.
