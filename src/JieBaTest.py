#coding=utf-8 
'''
Created on 2018年5月29日

@author: bingqiw
'''
import jieba
import jieba.posseg
# import sys
# reload(sys)
# sys.setdefaultencoding('gb2312')

if __name__ == '__main__':
    
    testSentence = "利用python进行数据分析"
    
#     print("1.精准模式分词结果："+"/".join(jieba.cut(testSentence,cut_all=False)))
#     print("2.全模式分词结果："+"/".join(jieba.cut(testSentence,cut_all=True)))
#     print("3.搜索引擎模式分词结果："+"/".join(jieba.cut_for_search(testSentence)))
#     print("4.默认（精准模式）分词结果："+"/".join(jieba.cut(testSentence)))

#     words = jieba.posseg.cut(testSentence)
#     for item in words:
#         print(item.word+"----"+item.flag)


    testSentence2=u"简书书院是一个很好的交流平台"
    print("+---------+---------+---------+---------+---------+---------+---------+----")
    print("1.加载词典前分词结果：")
    
#     for item in jieba.posseg.cut(testSentence2):
#         print(item.word+"---"+item.flag)
        
#     print([item for item in jieba.posseg.cut(testSentence2)])
    print("+---------+---------+---------+---------+---------+---------+---------+----")

    jieba.load_userdict("C:/Python27/Lib/site-packages/jieba/dict_my.txt")
    print("2.加载词典后分词结果：")
#     for item in jieba.posseg.cut(testSentence2):
#         print(item.word+"---"+item.flag)

    print("1.原始分词结果："+"/".join(jieba.cut("数据分析与数据挖掘的应用", HMM=False)))
    jieba.add_word("的应用")
    print("2.使用add_word(word, freq=None, tag=None)结果："+"/".join(jieba.cut("数据分析与数据挖掘的应用", HMM=False)))
    jieba.suggest_freq("的应用",tune=True)
    print("3.使用suggest_freq(segment, tune=True)结果："+"/".join(jieba.cut("数据分析与数据挖掘的应用", HMM=False)))



    
        