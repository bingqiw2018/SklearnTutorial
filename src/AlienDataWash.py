#coding=utf-8 
'''
Created on 2018年4月12日

@author: bingqiw
'''
import pandas as pd

def data_wash():
    
    name = 'D:/Program Files/eclipse4.6/workspace/SkLearnDemo/data/alien_data_500.xlsx'
    
    data_src = pd.read_excel(name,header=None,encoding='utf-8')
    df = pd.DataFrame(data_src)    
    print df.head()
    print df.shape
    df = df.loc[df[11] != df[4]] 
    print df.shape
    
    df.to_excel(name,header=None,index=False)
    
data_wash()