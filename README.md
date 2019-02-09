# interview 
The challenge comprises an NLP project.

## Use topics model with num_topics = 11 for ossible root cause

chlick here [for (Possible root cause](Possible_root_cause.ipynb)


## Use topics model with num_topics = 3 (not_good, bad, so_bad) for ossible root similar_Consumer 

Click here [for similar_Consumer](similar_Consumer(Consumer_complaint_narrative).ipynb)


A short report of project and reasult in [powerpint file](Closer_interview.pptx) **You should dmwoload it and see your computer**


## Api that are used for this project:

"""python

from nltk import word_tokenize  # for tokenize
import seaborn as sns  # for visualation
import matplotlib.pyplot as plt  # for visualation
from dask import delayed   # for multiprocessing and decreases the cost of running


from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel # topic models
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary    # dicationary for LDAModel
import pyLDAvis.gensim  #  for visualation LDA model.
import gensim

# My library: (for pre-prcessing  for text analysis)
from package.TextEditor import Remove_repetitive_words, Remove_stop_words, List_cleaner 
"""
