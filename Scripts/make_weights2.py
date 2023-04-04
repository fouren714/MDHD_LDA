from preprocess import preprocess
from LDA2 import LDA_model2
import pandas as pd
import numpy as np
import os
import pyLDAvis
from pathlib import Path
import sys
import os

class make_weights2:

    
    # Enter base path containing directory holding the corpus, the directory name, and how many ensembles should be trained.
    # saliency scores will be averaged from all iterations
    def __init__(self,base_path,directory,target_path,iterations,name,save,syn_path):
        self.base_path = base_path
        self.directory = directory
        self.target_path = Path(target_path)
        self.iterations = iterations
        self.name = name
        self.save = save
        self.syn_path = syn_path

    def make_model(self):
        self.model = LDA_model2(path_base = self.base_path,
                                folder = self.directory,
                                model_type = 'ensemble',
                                model_name = 'weight_maker_{}'.format(self.name),
                                syn_path = self.syn_path)

    
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
            # TESTING IF CORPUS BUILD MESSES THINGS UP --> we're good, but keep
            # in loop to get different synonym rules (maybe makes sense).
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

# pull weights from two dataframes to make a single dataframe
# compare industry and acadamic corpora in a single df
# saliency must be stored under "Normalized_Saliency" for each "Term"
# academic corpus is df1, industry corpus is df2
def concat_weights(df1,df2):

    key = ['long_term_cost','short_term_cost','policy_incentives',
        'suitability_compatibility','familiarity_knowledge','norms_attitudes',
        'brand_image','sustainability','reliability_uncertainty','driver_acceptance']
    
    def get_term_saliency(term,df):
        tmp_df = df.loc[df['Term'] == term]
        return tmp_df['Normalized_Saliency'].item()
    
    compare = {'Term':[],'Academic_Normalized_Saliency':[],'Industry_Normalized_Saliency':[]}
    for term in key:
        academic_saliency = get_term_saliency(term,df1)
        industry_saliency = get_term_saliency(term,df2)
        compare['Term'].append(term)
        compare['Academic_Normalized_Saliency'].append(academic_saliency)
        compare['Industry_Normalized_Saliency'].append(industry_saliency)
    return pd.DataFrame(compare)

# ***************** Verification Functions **********************************
def calculate_variance_pct(series):
    variance = series.var()
    variance_pct = (variance / series.min())*100
    return variance, variance_pct

def get_term_stats(df,term):
    tmp_df = df.loc[df['Term'] == term]
    mean = tmp_df['Normalized_Saliency'].mean()
    variance, pct = calculate_variance_pct(tmp_df['Normalized_Saliency'])
    return mean, variance, pct

def get_stats_df(academic_df,industry_df):
    key = ['long_term_cost','short_term_cost','policy_incentives',
    'suitability_compatibility','familiarity_knowledge','norms_attitudes',
    'brand_image','sustainability','reliability_uncertainty','driver_acceptance']

    industry_verification_dict = {'Term':[],'Normalized_Saliency':[],'Variance':[],'Percent_Variance':[]}
    academic_verification_dict = {'Term':[],'Normalized_Saliency':[],'Variance':[],'Percent_Variance':[]}
    for term in key:
        academic_mean, academic_variance, academic_variance_pct = get_term_stats(academic_df, term)
        industry_mean, industry_variance, industry_variance_pct = get_term_stats(industry_df, term)

        industry_verification_dict['Term'].append(term)
        industry_verification_dict['Normalized_Saliency'].append(industry_mean)
        industry_verification_dict['Variance'].append(industry_variance)
        industry_verification_dict['Percent_Variance'].append(industry_variance_pct)

        academic_verification_dict['Term'].append(term)
        academic_verification_dict['Normalized_Saliency'].append(academic_mean)
        academic_verification_dict['Variance'].append(academic_variance)
        academic_verification_dict['Percent_Variance'].append(academic_variance_pct)

    academic_stats_df = pd.DataFrame(academic_verification_dict)
    industry_stats_df = pd.DataFrame(industry_verification_dict)

    academic_final_saliency_df = academic_stats_df[['Term','Normalized_Saliency']].copy()
    industry_final_saliency_df = industry_stats_df[['Term','Normalized_Saliency']].copy()

    return academic_stats_df,industry_stats_df

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