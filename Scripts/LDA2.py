import pandas
import gensim
from txt_preprocess import *
import re
import spacy
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import preprocess2
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.interpolate import griddata
import pyLDAvis.gensim_models
import pickle
import pyLDAvis
import sys
import importlib.util

class LDA_model2:

    def __init__(self, path_base, folder, model_type, model_name, syn_path):
        self.path_base = path_base
        self.folder = folder
        self.model_type = model_type
        self.model_name = model_name
        self.syn_path = syn_path
    
    # def set_pyLDAvis_path(self):
    #     spec = importlib.util.spec_from_file_location("pyLDAvis","C:/Users/ouren/Documents/School_Local/eHDV_Adoption/pyLDAvis")
    #     imp = importlib.util.module_from_spec(spec)
    #     sys.modules["module.name"] = imp
    #     spec.loader.exec_module(imp)
    #     imp.LDA_model()

    
    def build_corpus(self):
        self.files = preprocess2.preprocess2(self.path_base,self.folder, self.syn_path)
        self.id2word, self.corpus, self.texts = self.files.build_corpus()
    
    

    def compute_coherence_values(self,corpus,dictionary,k,a,b):
        
        lda_model = gensim.models.LdaMulticore(corpus = corpus,
        id2word = dictionary, num_topics = k, random_state = 100, 
        chunksize = 100, passes = 10, alpha = a, eta = b)

        coherence_model_lda = CoherenceModel(model = lda_model, texts = self.texts, dictionary = self.id2word,
        coherence = 'c_v')

        return coherence_model_lda.get_coherence()
    
    def og_alpha(self):
        alpha = list(np.arange(0.01,1,0.5))
        alpha.append('symmetric')
        alpha.append('asymmetric')
        return alpha
    
    def og_beta(self):
        beta = list(np.arange(0.01,1,0.3))
        beta.append('symmetric')
        return beta

    def finetune(self,topic_bounds):
    
    # topic bounds is min and max topics in list format [min,max]
    # min topics > 1
        
        beta = self.og_beta()
        alpha = self.og_alpha()
        
        try:
            topics_range = list(range(topic_bounds[0],topic_bounds[1]+1))
        except:
            print('topic bounds is min and max topics in list format [min,max], min > 1')
            
        self.model_results = {'Validation_Set':[],'Topics':[],'Alpha':[],'Beta':[],'Coherence':[]}
        
        num_docs = len(self.corpus)
        corpus_sets = [gensim.utils.ClippedCorpus(self.corpus,int(num_docs*0.75))] # experiment from 0.25 - 0.75
        corpus_title = ['75% Corpus','100% corpus']
        
        # iterate over validation corpora
        for i in range(len(corpus_sets)):
            # iterate number of topics
            for k in topics_range:
                #alpha = get_alpha(k) use for alpha dependent on num_topics
                #iterate alpha:
                for a in alpha:
                    # iterate beta
                    for b in beta:
                        cv = self.compute_coherence_values(corpus = corpus_sets[i],dictionary = self.id2word, k = k,
                        a = a, b = b)

                        # Save Results
                        self.model_results['Validation_Set'].append(corpus_title[i])
                        self.model_results['Topics'].append(k)
                        self.model_results['Alpha'].append(a)
                        self.model_results['Beta'].append(b)
                        self.model_results['Coherence'].append(cv)
        
        self.df = pd.DataFrame(self.model_results)

        

    def plot_topics(self):
        
        self.df.plot.scatter(x = 'Topics',y='Coherence',xlabel='Topics',ylabel = 'Coherence')
    
    
    def quantize_alpha(self,alpha):
        if 'symmetric' == alpha:
            alpha = 1.0
        if 'asymmetric' == alpha:
            alpha = 1.1
        return alpha
    def quantize_beta(self,beta):
        if 'symmetric' == beta:
            beta = 1.0
        return beta
    
    def unqunatize(self,val):
        if val == 1.0:
            return 'symmetric'
        if val == 1.1:
            return 'asymmetric'
        else:
            return float(val)
    
    # number of topics that results in max coherence
    def get_max_vals(self):
        # make dataframe with all options at best number of topics
        max_topics = self.df.Topics.loc[self.df.Coherence == self.df.Coherence.max()].item()
        self.topic_df = self.df.loc[self.df.Topics == max_topics]

        self.topic_df['Beta'] = self.topic_df.apply(lambda x: self.quantize_beta(x['Beta']),axis=1)
        self.topic_df['Alpha'] = self.topic_df.apply(lambda x: self.quantize_alpha(x['Alpha']),axis=1)

        self.topic_df.Alpha = self.topic_df.Alpha.astype(float)
        self.topic_df.Beta = self.topic_df.Beta.astype(float)
        
        # get parameters that make highest coherence
        self.max_df = self.df.loc[self.df.Coherence == self.df.Coherence.max()]
        self.a = self.unqunatize(self.max_df.Alpha.item())
        self.b = self.unqunatize(self.max_df.Beta.item())
        self.k = self.unqunatize(self.max_df.Topics.item())
     
    def plot_ab(self):

        self.get_max_vals()

        x1 = np.linspace(self.topic_df['Alpha'].min(),self.topic_df['Alpha'].max(),len(self.topic_df['Alpha'].unique()))
        y1 = np.linspace(self.topic_df['Beta'].min(),self.topic_df['Beta'].max(),len(self.topic_df['Beta'].unique()))

        X,Y = np.meshgrid(x1,y1)
        Z = griddata((self.topic_df['Alpha'],self.topic_df['Beta']),self.topic_df['Coherence'],(X,Y),method = 'cubic')

        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        surf = ax.plot_surface(X,Y,Z,rstride = 1, cstride=1, cmap=cm.coolwarm,linewidth=0,antialiased=False)

        ax.set_zlim(0.3,0.5)
        ax.zaxis.set_major_locator(LinearLocator(10))

        ax.set_xlabel('Alpha')
        ax.set_ylabel('Beta')
        ax.set_zlabel('Coherence')
        ax.set_title('Coherence vs. Alpha vs. Beta @ Optimal Topic Num')                
    
    def build_model(self):
        
        if self.model_type == 'ensemble':
            print('Using Ensemble LDA')
            self.lda_model = gensim.models.EnsembleLda(corpus=self.corpus, id2word=self.id2word,
            random_state=100, passes = 3, ensemble_workers = 4, num_models = 8, distance_workers = 4)
        else:
            print(' Using LDA Multicore')
            self.get_max_vals() # requires a loaded sheet or running finetune
            self.lda_model = gensim.models.LdaMulticore(corpus=self.corpus, id2word=self.id2word, num_topics = self.k,
            alpha = self.a, eta = self.b,random_state=100, chunksize = 100, passes = 3, per_word_topics = True)

    def make_pyvis(self,save):
        #self.set_pyLDAvis_path()
        print('building model')
        self.build_model()
        print('model built')
        # pyLDAvis.enable_notebook()
        # print('notebook enabled')
        if self.model_type == 'ensemble':
            self.lda_model.num_topics = len(self.lda_model.stable_topics)
            self.lda_model.state = self.lda_model.classic_model_representation.state
        print('making pyLDAvis')
        self.LDAvis_prepared = pyLDAvis.gensim_models.prepare(self.lda_model,self.corpus,self.id2word,R = 100)
        
        if save == True:
            pyLDAvis.save_html(self.LDAvis_prepared,'{}lda.html'.format(self.model_name))

    def coherence_df(self):
        self.topics = self.lda_model.top_topics(self.corpus)
        
        # put Umass coherence scores in dataframe
        topic_dic = {'topic':[],'word':[],'score':[]}
        coherence_dic = {'topic':[],'UMass Coherence':[]}
        k = 1
        for topic in self.topics:
            #print('topic: ',topic)
            for lst in topic:
                #print('lst: ',lst)
                if type(lst) != np.float64:
                    for pair in lst:
                        #print('pair: ', pair)
                        topic_dic['score'].append(pair[0])
                        topic_dic['word'].append(pair[1])
                        topic_dic['topic'].append(k)
                        
                else:
                    coherence_dic['topic'].append(k)
                    coherence_dic['UMass Coherence'].append(lst)
            k += 1
        self.tops_df = pd.DataFrame(topic_dic)
        #tops_df.index = tops_df.score
        coherence_df = pd.DataFrame(coherence_dic)

        self.top4_df = self.tops_df[self.tops_df.topic <= 4]
        self.top4_df = self.top4_df.drop(columns = 'topic',axis=1)

    def saliency_df(self,model):
        df =  model.LDAvis_prepared.topic_info
        df = df.iloc[0:99]
        total = df.saliency.sum()
        df['norm_score'] = df.saliency / total
        df.reset_index(drop=True,inplace=True)
        
        return df
    
    def get_sum(self):

        self.coherence_df()
        sums = {'word':[],'score':[]}
        words = self.top4_df.word.unique()
        for word in words:
            tmp = self.top4_df[self.top4_df.word == word]
            score = tmp.score.sum()
            sums['word'].append(word)
            sums['score'].append(score)
        self.score_totals = pd.DataFrame(sums)
        total = self.score_totals.score.sum()
        self.score_totals['norm_score'] = self.score_totals.score / total

    def diff(self,v1,v2):
        return (v1 - v2)**2

    # compare scores takes 2 validation corpora and makes diff_df for the called object
    # difference is represented in sum squares form 
    def compare_scores(self,model_1,model_2): # models are the LDA model objects
        difference = {'Term':[],'difference':[]}
               
        df_1 = self.saliency_df(model_1)
        df_2 = self.saliency_df(model_2)
        
        for Term in df_1.Term:
            if df_2.Term.eq(Term).any():
                score1 = df_1.norm_score[df_1.Term == Term].item()
                score2 = df_2.norm_score[df_2.Term == Term].item()
                # print('score 1: ',score1)
                # print('score 2: ',score2)
                difference['Term'].append(Term)
                difference['difference'].append(self.diff(score1,score2))
        self.diff_df = pd.DataFrame(difference)
    
    def model_package(self):
        self.build_corpus()
        self.df = pd.read_csv('C:/Users/ouren/Documents/School_Local/eHDV_Adoption/LDA_Processing/notebooks/search_results_7_27B.csv')
        self.make_pyvis( save = False) 
    
    def compare_df(self,model_a,model_b, full_model_df):
        saliency = {'Term':[],'saliency':[],'difference':[]}
        df_a = model_a.LDAvis_prepared.topic_info.iloc[0:99]
        df_b = model_b.LDAvis_prepared.topic_info.iloc[0:99]

        df_a['norm_score'] = df_a.saliency / df_a.saliency.sum()
        df_b['norm_score'] = df_b.saliency / df_b.saliency.sum()

        for Term in self.diff_df.Term:
            if full_model_df.Term.eq(Term).any():
                # score_1 = full_model_df_a.norm_score[full_model_df_a.Term == Term].item()
                # score_2 = full_model_df_b.norm_score[full_model_df_b.Term == Term].item()
                score = full_model_df.norm_score[full_model_df.Term == Term].item()
                difference = self.diff_df.difference[self.diff_df.Term == Term].item()
                #print(score)
                saliency['saliency'].append(score)
                saliency['Term'].append(Term)
                saliency['difference'].append(difference)
                
        self.compare_saliency_df = pd.DataFrame(saliency)
        self.compare_saliency_df['difference_percent'] = (self.compare_saliency_df.difference / self.compare_saliency_df.saliency).astype(float)
        self.compare_saliency_df.difference = self.compare_saliency_df.difference.astype(float)

        self.key = ['long_term_cost','short_term_cost','policy_incentives',
        'suitability_compatibility','familiarity_knowledge','norms_attitudes',
        'brand_image','sustainability','reliability_uncertainty','driver_acceptance']
         
        self.compare_saliency_df = self.compare_saliency_df[self.compare_saliency_df['Term'].isin(self.key)]

    



if __name__ == "__main__":
    print("hi")
