# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 18:15:10 2018

@author: xfanu
"""

import operator
import csv 
import numpy as np 
import pandas as pd 

from random import randrange
from gensim import corpora, models 
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import pairwise_distances

def create_simi_matrix(randmax,input_path,dict_path,corpus_path,finalmatrix_path,evalmatrix_path):
    ################
    ################  change the input path here
    ################
    with open(input_path,'r',encoding="utf8") as csvfile: 
        reader = csv.reader(csvfile) 
        column_id = [row[0]for row in reader]

    with open(input_path,'r',encoding="utf8") as csvfile: 
        reader = csv.reader(csvfile) 
        column_pos = [row[12]for row in reader]

    with open(input_path,'r',encoding="utf8") as csvfile: 
        reader = csv.reader(csvfile) 
        column_product = [row[1]for row in reader]    

    with open(input_path,'r',encoding="utf8") as csvfile: 
        reader = csv.reader(csvfile) 
        column_brand = [row[3]for row in reader]    

    with open(input_path,'r',encoding="utf8") as csvfile: 
        reader = csv.reader(csvfile) 
        column_ingredient_list = [row[30:40]for row in reader]

    column_ingredient_list = column_ingredient_list[1:]
    column_ingredient = []  
    for ingredient in column_ingredient_list:
        ingredient = ",".join(ingredient)
        column_ingredient.append(ingredient)

    column_ingredient = [[ingredient.strip() for ingredient in column.lower().split(',')] for column in column_ingredient]

    dictionary = corpora.Dictionary(column_ingredient)

    ################ 
    ################  change dict path here
    ################
    dictionary.save(dict_path)  
    # store the dictionary, for future reference
    #com = [[a,b,c] for a,b,c in zip(column_id,column_pos,column_ingredient)]

    #print(dictionary.token2id)
    #######################testing#####################################
    ingredient_vec = [] 
    for column in column_ingredient: 
        ingredient_vec.append(dictionary.doc2bow(ingredient for ingredient in column))

    ################ 
    ################  change corpus path here
    ################
    corpora.MmCorpus.serialize(corpus_path, ingredient_vec)

    ingrecorpus = corpora.MmCorpus(corpus_path)

    ################ 
    ################  change the num of topic here
    ################
    lda = models.ldamodel.LdaModel(corpus=ingrecorpus, num_topics=50, iterations=1000, id2word = dictionary, passes=50)

    ##lda.print_topics(20)

    ###ingredient_vec[2800]


    #####################similarity matrix#######################
    ingredient_array = []
    for i in range(len(ingredient_vec)):
        ingredient_array.append(lda[ingredient_vec[i]])


    ingredient_lda_array = []
    for vec in ingredient_array:
        ################ 
        ################  change the num of topic here
        ################
        temp = [0]*50
        for (key,value) in vec:
            temp[key] = value
        ingredient_lda_array.append(temp)

    ingredient_lda_array = np.array(ingredient_lda_array)

    simi_matrix = 1 - pairwise_distances(ingredient_lda_array,metric="cosine")

    ##################final product similarity matrix###############
    final_matrix = []
    for i in range(len(simi_matrix)):
        a = simi_matrix[i].tolist()
        simi_dict = {}
        for i in range(len(a)):
            simi_dict[column_id[i+1],column_brand[i+1],column_product[i+1],column_pos[i+1]] = a[i]
        sorted_dict = sorted(simi_dict.items(), key=operator.itemgetter(1),reverse = True)    
        final_matrix.append(sorted_dict[0:11])

    ##################final product similarity matrix for evaluation###############
    eval_matrix = []
    for i in range(11):
        random_index = randrange(0,randmax)
        eval_matrix.append(final_matrix[random_index])
    ###################save similarity matrix######################


    ################ 
    ################  change the final matrix output path here
    ################
    csvfile = finalmatrix_path

    with open(csvfile, "w",encoding="utf-8") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(final_matrix)

    ################ 
    ################  change the final eval matrix output path here
    ################
    csvfile = evalmatrix_path

    with open(csvfile, "w",encoding="utf-8") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(eval_matrix)
    
    return eval_matrix