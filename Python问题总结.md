# Python基础语法
## JSON格式
### json.dump()与json.load()
* json.dump()用于接收两个对象一个是要写入的python数据，一个可用于存储数据的文件对象(`with open("data.txt",'w') as f: 中的f就是可用于存储数据的文件对象`)，用于将
python数据结构写入.json文件中
* json.load()只有一个对象就是json文件对象
### json.dumps()与json.loads()
* json.dumps()用于将python数据结构转换为json编码的字符串
* json.loads()用于将json编码的字符串转换为python数据结构
### 区别：
* 多一个s处理字符串(String),少一个s为处理文件
* [详见链接](https://www.cnblogs.com/everfight/p/json_file.html)
-------------
## 如何简单地理解Python中的if __name__ == '__main__'
* [详细链接](https://blog.csdn.net/yjk13703623757/article/details/77918633/)
-------------
## numpy库中的ndarray
### 例子
![图片有误](https://raw.githubusercontent.com/Ethan-1997/IT-/master/images/adarray.png)
-------------
# Python数据挖掘
## 亲和性分析
* 什么是亲和性分析
  > __亲和性分析根据样本个体（物体）之间的相似度，确定它们关系的亲疏__
* 一个亲和性分析的例子
  > 顾客在购买一件商品时，商家可以趁机了解，即顾客在购买一件商品时，商家可以趁机了解解他们还想买什么，
  以便把多数顾客愿意同时购买的商品放到一起销售以提升销售额。当商家收集到足够多的数据时，就可以对其进行亲和性分析，
  以确定哪些商品适合放在一起出售。
* 支持度与置信度
  >我们要找出“如果顾客购买了商品X，那么他们可能愿意购买商品Y”这样    
  的规则。简单粗暴的做法是，找出数据集中所有同时购买的两件商品。找出规则后，还需要判
  断其优劣，我们挑好的规则用。而支持度和置信度是常用的衡量规则优劣的方法
  >>* 支持度=顾客买了X然后又买Y应验的次数
  >>* 置信度=支持度/顾客买了X的应验次数
* 实战代码
```python
# coding: utf-8
#  NumPy（Numerical Python的缩写）是一个开源的Python科学计算库。使用NumPy，就可以很自然地使用数组和矩阵。
#  NumPy包含很多实用的数学函数，涵盖线性代数运算、傅里叶变换和随机数生成等功能
#  NumPy 通常与 SciPy（Scientific Python）和 Matplotlib（绘图库）一起使用
#  scikit-learn是用Python开发的机器学习库，它包含大量机器学习算法、数据集、工具和框架 在site-packet包中为sklearn
#  用命令pip3 install -U scikit-learn安装，安装时会自动安装nump、scipy

# In[1]:
  
import numpy as np # 导入第三方库numpy，别名为np 
dataset_filename = "affinity_dataset.txt" 
X = np.loadtxt(dataset_filename) # 导入当前路径下的数据集affinity_dataset.txt,X为ndarray类型
# shape：当X是一维时，表示元素个数   当X是二维时，表示二维数组行列数
# n_samples为行，此处为每个顾客买的商品情况，n_features为列，此处为商品类别
n_samples, n_features = X.shape
print("This dataset has {0} samples and {1} features".format(n_samples, n_features))


# In[2]:

# 打印X数组前五行，X[:5]用法详见Python语法
print(X[:5])


# In[3]:

# 商品设为bread milk cheese appies bananas
features = ["bread", "milk", "cheese", "apples", "bananas"]


# 第一个例子中计算如果顾客买了苹果，还买香蕉的支持度和置信度
# In[4]:

# 首先计算多少行包含了买苹果这个前提：
num_apple_purchases = 0
for sample in X:
    if sample[3] == 1:  # 如果这个顾客买了苹果
        num_apple_purchases += 1
print("{0} people bought Apples".format(num_apple_purchases)) # 已经计算出多少顾客买了商品


# In[5]:


# How many of the cases that a person bought Apples involved the people purchasing Bananas too?
# Record both cases where the rule is valid and is invalid.
rule_valid = 0
rule_invalid = 0
for sample in X:
    if sample[3] == 1:  # This person bought Apples
        if sample[4] == 1:
            # This person bought both Apples and Bananas
            rule_valid += 1
        else:
            # This person bought Apples, but not Bananas
            rule_invalid += 1
print("{0} cases of the rule being valid were discovered".format(rule_valid))
print("{0} cases of the rule being invalid were discovered".format(rule_invalid))


# In[6]:

# Now we have all the information needed to compute Support and Confidence
support = rule_valid 
# The Support is the number of times the rule is discovered.
confidence = rule_valid / num_apple_purchases
# Confidence can be thought of as a percentage using the following:
print("As a percentage, that is {0:.1f}%.".format(100 * confidence))


# In[7]:

from collections import defaultdict
# Now compute for all possible rules
valid_rules = defaultdict(int)
invalid_rules = defaultdict(int)
num_occurences = defaultdict(int)

for sample in X:
    for premise in range(n_features):
        if sample[premise] == 0: continue
        # Record that the premise was bought in another transaction
        num_occurences[premise] += 1
        for conclusion in range(n_features):
            if premise == conclusion:  # It makes little sense to measure if X -> X.
                continue
            if sample[conclusion] == 1:
                # This person also bought the conclusion item
                valid_rules[(premise, conclusion)] += 1
            else:
                # This person bought the premise, but not the conclusion
                invalid_rules[(premise, conclusion)] += 1
support = valid_rules # support此时和valid_rules指向同一个内存空间
confidence = defaultdict(float)
for premise, conclusion in valid_rules.keys(): # valid_rules。keys()为所有键值
    confidence[(premise, conclusion)] = valid_rules[(premise, conclusion)] / num_occurences[premise]


# In[8]:

for premise, conclusion in confidence: # confidenc时键为元组，值为float 所以键有两个元素值 premise与conclusion分别对应到了这两个值
    premise_name = features[premise] 
    conclusion_name = features[conclusion]
    print("Rule: If a person buys {0} they will also buy {1}".format(premise_name, conclusion_name))
    print(" - Confidence: {0:.3f}".format(confidence[(premise, conclusion)])) # .3f为保留小数后后三位
    print(" - Support: {0}".format(support[(premise, conclusion)]))
    print("")


# In[9]:
 # 将Int[8]的内容写成一个方法
def print_rule(premise, conclusion, support, confidence, features):
    premise_name = features[premise]
    conclusion_name = features[conclusion]
    print("Rule: If a person buys {0} they will also buy {1}".format(premise_name, conclusion_name))
    print(" - Confidence: {0:.3f}".format(confidence[(premise, conclusion)]))
    print(" - Support: {0}".format(support[(premise, conclusion)]))
    print("")


# In[10]:
# 测试Int[9]的方法
premise = 1
conclusion = 3
print_rule(premise, conclusion, support, confidence, features)


# In[11]:

# Sort by support
from pprint import pprint
pprint(list(support.items())) # Python 字典 items() 方法以列表返回可遍历的(键, 值) 元组数组。pprint使得打印结果更加美观


# In[12]:

from operator import itemgetter
sorted_support = sorted(support.items(), key=itemgetter(1), reverse=True) # itermgetter(1)表示支持度的字典的值而不是键，reverse=Ture表示逆序


# In[13]:

for index in range(5):
    print("Rule #{0}".format(index + 1))
    (premise, conclusion) = sorted_support[index][0] # sorted_support 为（((1,2),3), 这种类型，sorted_support[index][0]为键元组
                                                                          ((2,3),4),
                                                                          ((3,4),6))
    print_rule(premise, conclusion, support, confidence, features) # 根据支持度从高到底排序后打印规则


# In[14]:


sorted_confidence = sorted(confidence.items(), key=itemgetter(1), reverse=True)


# In[15]:

for index in range(5):
    print("Rule #{0}".format(index + 1))
    (premise, conclusion) = sorted_confidence[index][0]
    print_rule(premise, conclusion, support, confidence, features)  # 根据置信度从高到底排序后打印规则
```
## 根据iris数据集简单分类
### 
