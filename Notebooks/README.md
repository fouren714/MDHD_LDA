# Jupyter Notebooks for LDA Processing and Analysis

This directory contains Jupyter notebooks for preprocessing text data, generating topic models using Latent Dirichlet Allocation (LDA), and analyzing the results. Each notebook serves a specific purpose in the workflow.

## Notebooks

### 1. class_dev.ipynb
This notebook includes the development and testing of LDA models and preprocessing methods.

#### Key Sections:
- **Setup Models**: Initializes and sets up LDA models.
- **Validation**: Contains methods for validating the models.
- **Visualize Difference**: Visualizes the differences between models.
- **Code in Development**: Various code sections under development including saliency and KL divergence calculations.

### 2. industry_dev.ipynb
This notebook focuses on developing industry-specific LDA models and preprocessing techniques.

### 3. paper_sources.ipynb
This notebook is used for analyzing academic sources and visualizing the data.

#### Key Sections:
- **Academic Sources**: Counts and visualizes the number of papers.
- **Visualizations**: Includes visualizations for academic and industry sources.

### 4. sandbox.ipynb
This notebook is used for testing and validating various preprocessing and modeling techniques.

#### Key Sections:
- **Make weights testing**: Tests the weight generation process.
- **Model Guts**: Details of the model internals.
- **Weight Concatenation**: Combines weights from different sources.
- **Visualizations**: Includes various visualizations such as radar and bar charts.

### 5. synonym_builder_tst.ipynb
This notebook tests the synonym building process and its impact on LDA models.

#### Key Sections:
- **Load Full List**: Loads the full list of synonyms.
- **Random Selection**: Randomly selects synonyms for categories.
- **Apply New Synonym Rules**: Tests new synonym rules.
- **Build Models**: Builds models with different synonym rules.
- **Analysis**: Analyzes the results using various visualizations.

## Getting Started

### Prerequisites
- Python 3.x
- Required Python libraries: `pandas`, `numpy`, `gensim`, `nltk`, `pyLDAvis`, `plotly`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repository.git
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
Open the desired notebook in Jupyter Notebook or JupyterLab and run the cells to execute the code. Make sure to set up the necessary paths and data files as required by each notebook.

## Contributing
Feel free to fork this repository and make contributions. Pull requests are welcome.

## License
This project is licensed under the MIT License.
