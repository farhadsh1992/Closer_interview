# interview 
The challenge comprises an NLP project.

## Use topics model with num_topics = 11 for possible root cause

chlick here [for Possible root cause](Possible_root_cause.ipynb)

**pre_processing for text:**
1. Remove stop_words,
2. Add Product column to Nan Sub_product column, 
3. Add Issue column to Nan Sub_Issue column, (for improve length of every row )

**Model:**
- find the best num_topics and n_grams model by draw chorsnce plot.
- Choose 1_grams and 2_grams, and num_topics=11

## Use topics model with num_topics = 3 (not_good, bad, so_bad) for ossible root similar_Consumer 

Click here [for similar_Consumer](similar_Consumer(Consumer_complaint_narrative).ipynb)

**pre_processing for text:**
1. Remove stop_words,
2. Replace number instead of  xxx xxx, 
3. remove repetitious words from every rows, (for improve length of every row )

**Model:**
- find the best num_topics and n_grams model by draw chorsnce plot.
- Choose 1_grams and 2_grams, and num_topics=3 (not_good, bad, so_bad)


<font color='red'>A short report of project and reasult in [powerpint file](Closer_interview.pptx) **You should dmwoload it and see your computer**</font>


## Apis are used for this project:

```python

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
```
