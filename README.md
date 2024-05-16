# MDHD_LDA
A sample repository showing work approved for sharing on Medium- and Heavy-Duty battery electric vehicle adoption using latent Dirichlet allocation (LDA). 
# Publication
[Developing a Profile of Medium- and Heavy-Duty Electric Vehicle Fleet Adopters with Text Mining and Machine Learning](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4373106)
## Notebooks
Contains workspaces that are used to run tests and develop new functions.

## Scripts
Contains scripts written to build LDA model and preprocess documents. 
### Preprocess/Preprocess2
- Functionality: preprocesses unstructured text in preparation for topic modeling with LDA.
- Inputs:
  - path_base: path to project directory where pdf files are stored.
  - folder: name of directory where pdf documents (corpus) are stored.
- Outputs
  - id2word vectorization of corpus and corpus formatted for LDA.

### LDA/LDA2
- Functionality: search for optimal LDA parameters, visualize grid space search, implement ensemble LDA, visualize LDA model with pyLDAvis.
  - NOTE: LDA2 is called by make_weights2.
- Inputs:
  - path_base: should be the same as that used in preprocess
  - folder: same as preprocess
  - model_type: 'ensemble' model type will trigger eLDA implementation and skip space search methods.
  - syn_path: path to synonym map. See format example in data directory.
- Outputs:
  - compare_saliency_df: DataFrame holding saliency results and validation statistics.

### make_weights2
- Functionality: Runs LDA(2) for multiple coropora. In our instance, this was academic and industry. Can implement on the command line.
- Inputs:
  - base_path: same as above
  - directory - equivalent to "folder" above
  - target_path: saving directory
  - iterations: determines how many times models should be fit
  - name: name to save the model by
  - save: boolean to determine saving pyLDAvis visualization
  - syn_path: same as above
- Outputs:
  - academic_stats_df/industry_stats_df: gives saliency for terms defined in get_stats_df method. These terms should be the same as those in the syn_map
  - academic_final_saliency_df/industry_final_saliency_df: gives normalized versions of term saliency.

# Data
- syn_map: format example for synonym mapping that is explained in publication.

