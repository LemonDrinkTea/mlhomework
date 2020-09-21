import os
import numpy as np
import string
import time
import math
def getFilePathList():
    filePath_listtrain = []
    filePath_listtest = []
    for walk in os.walk('train'):
        part_filePath_list = [os.path.join(walk[0], file) for file in walk[2]]
        filePath_listtrain.extend(part_filePath_list)
    for walk in os.walk('test'):
        part_filePath_list = [os.path.join(walk[0], file) for file in walk[2]]
        filePath_listtest.extend(part_filePath_list)
    return filePath_listtrain, filePath_listtest

#多项式
def loadData_m(filePath_list):
    classVec = []
    dataset=[]
    classword=[]
    classtext=[]
    vecaburary=set()
    for j in range(len(filePath_list)):
        classMap={}
        textcount=0
        f1 = open(filePath_list[j], "r", encoding='utf-8')
        stop = [line.strip() for line in open("stop_words_zh.txt", 'r', encoding='utf-8').readlines()]
        lines = f1.readlines()
        for i in range(0, lines.__len__(), 1):  # (开始/左边界, 结束/右边界, 步长)
            countMap = {}
            for word in lines[i].split():
                word = word.strip(string.whitespace)
                if word not in stop:

                    vecaburary.add(word)
                    if countMap.__contains__(word):
                        countMap[word] += 1
                    else:
                        countMap[word] = 1
                    if classMap.__contains__(word):
                        classMap[word] += 1
                    else:
                        classMap[word] = 1
            if countMap.__len__() != 0:
                textcount+=1
                classVec.append(j)
                dataset.append(countMap)
        classtext.append(textcount)
        classword.append(classMap)
    return classVec,dataset,classword,classtext,vecaburary



# 多项式模型
def cal_prob_m(classword, classtext,vecaburary):
    pnums=[]
    for i in range(6):
        pnums.append(sum(classword[i].values()) + len(vecaburary))
        for key in classword[i].keys():
            classword[i][key]=np.log((classword[i][key]+1.)/ pnums[i])
    category = [0.]*6
    for i in range(6):
        category[i] = np.log(classtext[i]/ sum(classtext))
    return category,classword,pnums


#多项式分类
def naive_byes_classify_m(category, classword,pnums,dataset,vecaburary):
    test_class = [0]*len(dataset)  # 先假设样本是一个全为0的向量
    for i,ve in enumerate(dataset):
        test = [0.] * 6
        for k in range(6):

            test[k]+=category[k]
            for j in ve.keys():
                if j in classword[k].keys():
                    test[k] += classword[k][j]*ve[j]  # 代入计算属于每一类别的概率
                else:
                    if j in vecaburary:
                        test[k] += np.log(1/pnums[k])*ve[j]
        test_class[i]=test.index(max(test))
    return test_class

#伯努利
def loadData(filePath_list):

    classVec = []
    classtext=[]
    vecaburary=set()
    x=[]
    for j in range(len(filePath_list)):
        classMap={}
        textcount=0
        f1 = open(filePath_list[j], "r", encoding='utf-8')
        stop = [line.strip() for line in open("stop_words_zh.txt", 'r', encoding='utf-8').readlines()]
        lines = f1.readlines()
        for i in range(0, lines.__len__(), 1):  # (开始/左边界, 结束/右边界, 步长)
            xx=[]

            for word in lines[i].split():
                word = word.strip(string.whitespace)
                if word not in stop:
                    vecaburary.add(word)
                    xx.append(word)
            if xx.__len__() != 0:
                x.append(xx)
                textcount+=1
                classVec.append(j)
        classtext.append(textcount)
    return classVec,x,classtext,vecaburary



# 伯努利模型
def cal_prob(dataset,classtext,vecaburary,classVec):
    classword = [{}.fromkeys(vecaburary, 0) for _ in range(6)]
    for i,data in enumerate(dataset):
        for word in set(data):
            classword[classVec[i]][word]+=1
    for i in range(6):
        classword[i] = dict(zip(classword[i].keys(), [(v + 0.1) / (classtext[i] + 0.2) for v in classword[i].values()]))
    category = [0.]*6
    for i in range(6):
        category[i] = np.log(classtext[i]/ sum(classtext))
    return category,classword
#伯努利分类
def naive_byes_classify(category,classword,dataset,vecaburary):
    test_class =[] # 先假设样本是一个全为0的向量
    vecaburary = {}.fromkeys(vecaburary)
    for ve in dataset:
        ve ={}.fromkeys(ve)
        tests=[]
        for k,p in enumerate(classword):
            test = 0
            test+=category[k]
            for j in vecaburary.keys():
                if j in ve:
                    test += math.log(p[j])  # 代入计算属于每一类别的概率
                else:
                    test += math.log(1-p[j])
            tests.append(test)
            # do argmax to get prediction label
        test_class.append(tests.index(max(tests)))
    return test_class

time_start=time.time()
filePath_listtrain, filePath_listtest = getFilePathList()
classVec, dataset , classword , classtext,vecaburary1 = loadData_m(filePath_listtrain)
category, classword,pnums=cal_prob_m(classword, classtext,vecaburary1)
classVec, dataset , classword1 , classtext,vecaburary = loadData_m(filePath_listtest)
result2=naive_byes_classify_m(category, classword,pnums,dataset,vecaburary1)
right=0
for i in range(len(result2)):
    if result2[i]==classVec[i]:
        right+=1
print('多项式测试集')
print('%.4f' %(right/len(result2)*1.00))




time_start=time.time()
classVec,dataset,classtext,vecaburary = loadData(filePath_listtrain)
category,classword=cal_prob(dataset,classtext,vecaburary,classVec)
classVec,dataset,classtext,vecaburary2 = loadData(filePath_listtest)
result2=naive_byes_classify(category,classword,dataset,vecaburary)
right=0
for i in range(len(result2)):
    if result2[i]==classVec[i]:
        right+=1
print('伯努利测试集')
print('%.4f' %(right/len(result2)*1.00))
time_end=time.time()
print('totally cost',time_end-time_start)