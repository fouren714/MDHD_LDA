from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
import os
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import pandas as pd
import gensim.corpora as corpora
import re
import spacy
import math
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
import nltk
nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')
import sys

class preprocess:

    def __init__(self, path_base, folder):
        self.path_base = path_base
        self.folder = folder
    
    
    # path is directory path to papers
    def get_filepaths(self):
   
        self.file_paths = []  # List which will store all of the full filepaths.

        file_location = '{}{}'.format(self.path_base,self.folder)
        # Walk the tree.
        for root, directories, files in os.walk(file_location):
            for filename in files:
                # Join the two strings in order to form the full filepath.
                filepath = os.path.join(root, filename)
                self.file_paths.append(filepath)  # Add it to the list.

        return self.file_paths 
    
    def pdf_to_txt(self,path):
        
        rsrcmgr = PDFResourceManager()
        retstr = StringIO()
        codec = 'utf-8'
        laparams = LAParams()
        device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
        
        
        fp = open(path, 'rb')
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        password = ""
        maxpages = 0
        caching = True
        pagenos=set()

        for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
            interpreter.process_page(page)

        text = retstr.getvalue()
        
        fp.close()
        device.close()
        retstr.close()
        return text

    ############## build_corpus helpers #####################################################

    def sent_to_words(self,sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence),deacc=True))

    
    # used for stopwords addon
    def get_excel(self):
        path = '{}stopwords_addon.xlsx'.format(self.path_base)
        df = pd.read_excel(path , sheet_name=0) 
        return df['adopt'].tolist()

    # def remove_stopwords(self):
    #     lst = []
    #     for doc in self.data_words:
    #         for word in doc:
    #             if word in self.stop_words:
    #                 continue
    #             else:
    #                 lst.append(word)
    #     return lst
    
    def remove_stopwords(self): # simple_preprocess(str(doc))
        return [[word for word in doc if word not in self.stop_words] for doc in self.data_words]
        print('stopwords removed')
    def remove_stopwords2(self): # simple_preprocess(str(doc))
        return [[word for word in doc if word not in self.stop_words] for doc in self.lemmatized]
        print('stopwords removed')
    def make_bigrams(self):
        self.bigram = gensim.models.Phrases(self.data_words,min_count=5, threshold=50)
        bigram_mod = gensim.models.phrases.Phraser(self.bigram)
        return [bigram_mod[doc] for doc in self.no_stops]
        print('bigrams created')

    def make_trigrams(self):
        trigram = gensim.models.Phrases(self.bigram[self.data_words],threshold=50)
        trigram_mod = gensim.models.phrases.Phraser(trigram)
        return [trigram_mod[doc] for doc in self.bigrams]
        print('trigrams created')

    def lemmatization(self, allowed_postags = ['NOUN','ADJ','VERB','ADV']):
        texts_out = []
        nlp = spacy.load("en_core_web_sm",disable=['parser','ner'])
        for sent in self.trigrams:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    def process(self):
        self.no_stops = self.remove_stopwords()

        # syn_path = '{}synonyms2.xlsx'.format(path_base)
        # syn_words = syn(no_stops, syn_path, 1)

        self.bigrams = self.make_bigrams()
        self.trigrams = self.make_trigrams()
        self.lemmatized = self.lemmatization( allowed_postags=['NOUN','ADJ','VERB','ADV'])
        self.final_words = self.remove_stopwords2()
        return self.final_words

    def merge(self,series):
        return (series["word (in order of signifigance)"],series['synonym'])

    def syn(self): # path to synonyms excel sheet, sheets start at 0
        path = "{}synonyms3.xlsx".format(self.path_base)
        syns_df = pd.read_excel(path, sheet_name = 1)
        syns_df['pairs'] = syns_df.apply(self.merge, axis=1 )
        pairs = syns_df['pairs'].tolist()
        syns = syns_df['synonym'].unique().tolist()

        # get rid of irregular chars
        funky_f = chr(64258)
        funky_fleet = '{}eet'.format(funky_f)
        try:
            syns.remove(funky_fleet)
        except:
            print('No Funky Fleet Found')
            
        
        for i in range(len(self.lemmatized)):
            for j in range(len(self.lemmatized[i])):
                for k in range(len(pairs)):
                    if self.lemmatized[i][j] == pairs[k][0]:
                        self.lemmatized[i][j] = pairs[k][1]
        
        removed = []
        failed = []
        
        for i in range(len(self.lemmatized)):
            for j in range(len(self.lemmatized[i])):
                word = self.lemmatized[i][j]
                if word in syns:
                    continue
                else:
                    try:
                        self.lemmatized.remove(word)
                    except:
                        continue 
        
        return self.lemmatized

    def build_corpus(self):

        self.files = self.get_filepaths()
        
        #path_base = 'C:/Users/ouren/Documents/School_Local/eHDV_Adoption/LDA_Processing/notebooks/'
        
        self.corpus = []

        for file in self.files:
            text = self.pdf_to_txt(file)
            self.corpus.append(text)

        for entry in self.corpus:
            entry = entry.replace("\n"," ")
        
        self.data_words = list(self.sent_to_words(self.corpus))

        # Stop words
        add_on = self.get_excel()
        self.stop_words = stopwords.words('english')
        self.stop_words.extend(add_on)

        funky_f = chr(64258)
        funky_fleet = '{}eet'.format(funky_f)

        self.stop_words.extend([funky_fleet])

        # remove stop words, make bi/trigrams, lemmatize
        self.lemmatized = self.process()

        #Synonym work
        # syns_df = pd.read_excel('{}synonyms2.xlsx'.format(path_base),sheet_name=1)
        # syns_df['pairs'] = syns_df.apply(merge, axis=1 )
        # pairs = syns_df['pairs'].tolist()
        # syns = syns_df['synonym'].unique().tolist()

        # exchange words for synonyms
        syn_words = self.syn()

        #dict
        self.id2word = corpora.Dictionary(syn_words)

        texts = syn_words

        #Term document freqency
        self.corpus = [self.id2word.doc2bow(text) for text in texts]

        return self.id2word, self.corpus, texts

    #def save_docs(self):
        


#######################################################################################

if __name__ == "__main__":
    files = preprocess('C:/Users/ouren/Documents/School_Local/eHDV_Adoption/LDA_Processing/notebooks/')
    id2word , corpus = files.build_corpus()
    #print(corpus)
