from preprocess import preprocess
from LDA import LDA_model
import pandas as pd
import numpy as np
import os
import pyLDAvis
from pathlib import Path
import sys
import os

class make_weights:

    
    # Enter base path containing directory holding the corpus, the directory name, and how many ensembles should be trained.
    # saliency scores will be averaged from all iterations
    def __init__(self,base_path,directory,target_path,iterations,name,save):
        self.base_path = base_path
        self.directory = directory
        self.target_path = Path(target_path)
        self.iterations = iterations
        self.name = name
        self.save = save

    def make_model(self):
        self.model = LDA_model(path_base = self.base_path,
                                folder = self.directory,
                                model_type = 'ensemble',
                                model_name = 'weight_maker_{}'.format(self.name))

    
    # regardless of iterations, build corpus and pyvis once with model_package(self) from LDA_model
    def build_corpus(self):
        self.model.model_package()

    def create_weights(self):
        # outputs will go here
        self.saliency_df = pd.DataFrame(columns=['Term','Freq','Total','Category','logprob','loglift',
                                         'saliency','Normalized_Saliency'])

        # build model object and corpus
        self.make_model()
        
        # BUILD CORPUS USUALLY GOES HERE

        # create as many models as specified by iterations, log resutls
        k=0 

        final_scores = {'Term':[],'Normalized_Saliency':[]}
        for i in range(self.iterations):
            # TESTING IF CORPUS BUILD MESSES THINGS UP
            self.build_corpus()
            # rebuild  with saliency scores
            self.model.make_pyvis(save = False)
            # LDAvis prepared is customized to retain saliency, will not work with regular LDAvis 
            # package
            df = self.model.LDAvis_prepared.topic_info.iloc[0:99]

            #crop to relevant terms
            self.key = ['long_term_cost','short_term_cost','policy_incentives',
            'suitability_compatibility','familiarity_knowledge','norms_attitudes',
            'brand_image','sustainability','reliability_uncertainty','driver_acceptance']

            df = df[df['Term'].isin(self.key)]

            # Normalize scores. There are no negative saliency values so we can divide by max
            df['Normalized_Saliency'] = df['saliency'] / df['saliency'].max()

            set = [self.saliency_df,df]

            self.saliency_df = pd.concat(set)
            k+=1
            print('{} iterations complete'.format(k))
        
        # now that we have saliency for each iteration let's get the average saliency overall iterations
        # for each term

        for term in self.key:
            temp_df = self.saliency_df.loc[self.saliency_df['Term']==term]
            avg_normalized_saliency = temp_df['Normalized_Saliency'].mean()
            final_scores['Term'].append(term)
            final_scores['Normalized_Saliency'].append(avg_normalized_saliency)
        
        self.final_saliency_df = pd.DataFrame(final_scores)

        if self.save:
            self.final_saliency_df.to_csv(self.target_path / '{}_weights.csv'.format(self.name))

if __name__ == "__main__":

    BASE_PATH = sys.argv[1].replace(os.sep,'/')
    DIRECTORY = sys.argv[2]
    TARGET_PATH = sys.argv[3].replace(os.sep,'/')
    ITERATIONS = int(sys.argv[4])
    
    weight_maker = make_weights(base_path = BASE_PATH,
                                directory = DIRECTORY,
                                target_path = TARGET_PATH,
                                iterations = ITERATIONS)
    
    print('Base Path: {}\nDirectory: {}\nTarget Path: {}'.format(BASE_PATH,DIRECTORY,TARGET_PATH))
    
    weight_maker.create_weights()