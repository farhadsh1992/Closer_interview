#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 13:17:02 2019

@author: Farhad
For: Closer_interview
Title: banckComplaint

"""



from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
from nltk.tokenize import WordPunctTokenizer
import string
import re
from bs4 import BeautifulSoup
from nltk.stem import SnowballStemmer
import sys


def EstimateFaster(num,xlist, description):
    num+=1
    run = ("["+str(num)+'/'+str(len(xlist))+"]["+str(description)+']')
        
    sys.stdout.write('\r'+ run)


def Remove_stop_words(list_text,title='Remove stop_words'):
    
    stop_words = [x for x in stopwords.words('english') if x!='not'] 
    new_list = []
    
    for num,text in enumerate(list_text):
        filtered = []
        text_token = word_tokenize(str(text))
        for w in text_token:
            if w not in stop_words:
                filtered.append(w)
        EstimateFaster(num,list_text,title)
        new_list.append(" ".join(filtered[:]))
    print("*** Done! ***")   
    return new_list

def Remove_repetitive_words(old_list, title='Remove repetitive words'):
    new_list =[]
    for num,text in enumerate(old_list):
        ulist = []
        words = word_tokenize(text)
        [ulist.append(x) for x in words if x not in ulist]
        new_list.append((' '.join(ulist)).lower())
        EstimateFaster(num,old_list,title)
    print("*** Done! ***")
    return new_list



def Text_cleaner(text):
 
    stem = SnowballStemmer('english')
        
    tok = WordPunctTokenizer()
    number = r'@[0-9_]+'
    pat1 = r'https?://[^ ]+'
    www_pat = r'www.[^ ]+'
    x2_pat = r'xx'
    x3_pat = r'xxx'
    x4_pat= r'xxxx'
    card_pat= r'xxxx xxxx'
    
    part3 = string.punctuation # remove 's
    combined_pat = r'|'.join((number,pat1,x2_pat, x3_pat,x4_pat ))

    negations_dic = {"isn't":"is not", "aren't":"are not", 
                     "wasn't":"was not", "weren't":"were not",
                     "haven't":"have not","hasn't":"has not",
                     "hadn't":"had not","won't":"will not",
                     "wouldn't":"would not", "don't":"do not",
                     "doesn't":"does not","didn't":"did not",
                     "can't":"can not","couldn't":"could not",
                     "shouldn't":"should not","mightn't":"might not",
                     "mustn't":"must not","isnt":"is not", "arent":"are not", 
                     "wasnt":"was not", "werent":"were not",
                     "havent":"have not","hasnt":"has not",
                     "hadnt":"had not","wont":"will not",
                     "wouldnt":"would not", "dont":"do not",
                     "doesnt":"does not","didnt":"did not",
                     "cant":"can not","couldnt":"could not",
                     "shouldnt":"should not","mightnt":"might not",
                     "mustnt":"must not","ist":"is not", "aret":"are not", 
                          
                     "havet":"have not","hasnt":"has not",
                     "hadnt":"had not","wont":"will not",
                     "wouldt":"would not", "dont":"do not",
                     "doest":"does not","didt":"did not",
                     "cant":"can not","couldnt":"could not",
                     "shouldt":"should not"}
    neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')
        
        
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
        
    try:
        bom_removed = souped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        bom_removed = souped
            
    lower_case = bom_removed.lower()
    lower_case = stem.stem(lower_case)
    stripped = re.sub(www_pat, '', lower_case)
    stripped = re.sub(card_pat, 'card_number', stripped)  
    stripped = re.sub(combined_pat, '', stripped)
      
        
    neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], stripped)
    letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
    # During the letters_only process two lines above, it has created unnecessay white spaces,
    # I will tokenize and join together to remove unneccessary white spaces
    words = [x for x  in tok.tokenize(letters_only) if len(x) > 1]
        
    return (" ".join(words)).strip()




def List_cleaner(old_list, title='Clean every text'):
    new_list = []
    for num,text in enumerate(old_list):
        new_list.append(Text_cleaner(text))
        EstimateFaster(num,old_list,title)
    print("-- Done! --")
    return new_list

