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
------
## 根据iris数据集简单分类
### 分类应用目标：分类应用的目标是，根据已知类别的数据集，经过训练得到一个分类模型，再用模型对类别未知的数据进行分类
### Iris植物分类数据集
    * 共有150条植物数据
    * 每个数据给出四个特征：sepal length、sepal width、petal length、petal width（分别表示萼片和花瓣的长与宽），单位均为cm
    * 共有三种类别：Iris Setosa（山鸢尾）、Iris Versicolour（变色鸢尾）和Iris Virginica（维吉尼亚鸢尾）
### OneR算法
   > OneR算法的思路很简单，它根据已有数据中，具有相同特征值的个体最可能属于哪个类别进行分类
   * 举例说明：假如数据集的某一个特征可以取0或1两个值。数据集共有三个类别。特征值为0的情况下，A类有20个这样的个体，B类有60个，C类也有20个。那么特征值      为0的个体最可能属于B类，当然还有40个个体确实是特征值为0，但是它们不属于B类。将特征值为0的个体分到B类的错误率就是40%，因为有40个这样的个体分别属于A类和C类。特征值为1时，计算方法类似，不再赘述；其他各特征值最可能属于的类别及错误率的计算方法也一样。
### 根据OneR算法选取错误率最低的一个特征进行分类
   * 实战代码
```python
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# The OneR algorithm is quite simple but can be quite effective, showing the power of using even basic statistics in many applications.
# The algorithm is:
# 
# * For each variable
#     * For each value of the variable
#         * The prediction based on this variable goes the most frequent class
#         * Compute the error of this prediction
#     * Sum the prediction errors for all values of the variable
# * Use the variable with the lowest error

# In[2]:

# Load our dataset
from sklearn.datasets import load_iris #从datasets中导入数据集iris
#X, y = np.loadtxt("X_classification.txt"), np.loadtxt("y_classification.txt")
dataset = load_iris()
#print(type(dataset))
X = dataset.data # iris中的所有数据 类型为ndarray（有点像：每行为一个元素，这个元素为一个列表，总的一个列表，但列表之间无逗号分隔）
# print(X)
y = dataset.target # iris的分类标准 0 1 2 存储在一个列表中
print(y)
# print(type(X))
# print(dataset.DESCR)
# shape表示ndarray中元素的个数（一维时），表示行列数（二维时） n_smaples对应行数-->样本数，n_features对应列数-->样本特征数
n_samples, n_features = X.shape 
print(n_features) # 4


# Our attributes are continuous, while we want categorical features to use OneR. We will perform a *preprocessing* step called discretisation（离散化). At this stage, we will perform a simple procedure: compute the mean and determine whether a value is above or below the mean.

# In[3]:

# Compute the mean for each attribute
attribute_means = X.mean(axis=0) # axis=0时按行压缩求均值，X最后只剩一行，这一行有四列，对应iris的四个属性，attribute_means为ndarray
# mean()函数用法见附1
print(attribute_means)
# assert为断言 用法见附2 shape 用法见附3
assert attribute_means.shape == (n_features,) # 给attribute_means的shape赋值，因为只有一行四列，所以形式为（n_features,)
# print(attribute_means.shape) # 为(4,)
X_d = np.array(X >= attribute_means, dtype='int') # # 与平均值比较,大于等于的为“1”,小于的为“0”.将连续性的特征值变为离散性的类别型。 
print(X_d)


# In[4]:


# Now, we split into（分成） a training and test set
# cross_validation模块在0.18版本中被弃用，现在已经被model_selection代替。所以在导入的时候把"sklearn.cross_validation import  train_test_split "更改为
# "from sklearn.model_selection import  train_test_split"
from sklearn.model_selection import train_test_split

# Set the random state to the same number to get the same results as in the book
random_state = 14
# train_data：所要划分的样本特征集
# train_target：所要划分的样本结果
# test_size：样本占比，如果是整数的话就是样本的数量
# random_state：是随机数的种子。

# 训练集Xd_train和测试集Xd_test。
#y_train和y_test分别为以上两个数据集的类别信息
# train_test_split()函数用法见 附4
X_train, X_test, y_train, y_test = train_test_split(X_d, y, random_state=random_state)
print("There are {} training samples".format(y_train))
print("There are {} training samples".format(y_train.shape))

print("There are {} testing samples".format(y_test.shape))
print(X_train)
print(y_train)


# In[5]:


from collections import defaultdict
from operator import itemgetter


def train(X, y_true, feature):
    """Computes the predictors and error for a given feature using the OneR algorithm
    
    Parameters
    ----------
    X: array [n_samples, n_features]
        The two dimensional array that holds the dataset. Each row is a sample, each column
        is a feature.
    
    y_true: array [n_samples,]
        The one dimensional array that holds the class values. Corresponds to X, such that
        y_true[i] is the class value for sample X[i].
    
    feature: int
        An integer corresponding to the index of the variable we wish to test.
        0 <= variable < n_features
        
    Returns
    -------
    predictors: dictionary of tuples: (value, prediction)
        For each item in the array, if the variable has a given value, make the given prediction.
    
    error: float
        The ratio of training data that this rule incorrectly predicts.
    """
    # Check that variable is a valid number
    n_samples, n_features = X.shape # n_samples为数组行数，n_features为数组列数（即为特征个数）
    assert 0 <= feature < n_features # 判断特征的索引值是否在特征值范围内
    # Get all of the unique values that this variable has
    # X[:,feature]赋实参后为X_train[:,feature](第feature列X的训练集的特征值)
    # [:,feature]表示所有行的第feature列（列序号从0开始） set函数为求集合（注意集合三大特性，因为不重复性，所以values只有0和1）
    values = set(X[:,feature]) 
    # Stores the predictors array that is returned
    predictors = dict() # 创建一个空字典
    errors = []
    for current_value in values: # 遍历转换为0 1后的特征值列values
                                                        # X_train,y_train,variable(第几列特征),current_value选定列的当前某一个特定特征值
        most_frequent_class, error = train_feature_value(X, y_true, feature, current_value)
        predictors[current_value] = most_frequent_class
        errors.append(error)
    # Compute the total error of using this feature to classify on
    total_error = sum(errors)
    return predictors, total_error

# Compute what our predictors say each sample is based on its value
#y_predicted = np.array([predictors[sample[feature]] for sample in X])
  
# train_feature_value函数为计算每一列特征值（0和1）分别对应类别最准确的一个
                        # X_train,y_train,variable(第几列特征),current_value选定列的当前某一个特定特征值
def train_feature_value(X, y_true, feature, value):# 参数分别为数据集，类别数组，选好的特征索引值，特征值
    # Create a simple dictionary to count how frequency they give certain predictions
    class_counts = defaultdict(int) # defaultdict在索引值不存在时 返回对应类型的默认值
    # Iterate through each sample and count the frequency of each class/value pair
    for sample, y in zip(X, y_true):# zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的对象，在前面加list()即可转换为列表对象
        if sample[feature] == value:
            class_counts[y] += 1
    # Now get the best one by sorting (highest first) and choosing the first item
    sorted_class_counts = sorted(class_counts.items(), key=itemgetter(1), reverse=True)
    most_frequent_class = sorted_class_counts[0][0]
    # The error is the number of samples that do not classify as the most frequent class
    # *and* have the feature value.
    
    n_samples = X.shape[1] # X.shape[1]表示数据的列数
    error = sum([class_count for class_value, class_count in class_counts.items()
                 if class_value != most_frequent_class])
    return most_frequent_class, error


# In[6]:


# Compute all of the predictors
# variable表示特征值序号（即X的列数 0到3）X_train.shape具有行和列 是二维 shape[1]表示列-->即为属性个数
all_predictors = {variable: train(X_train, y_train, variable) for variable in range(X_train.shape[1])}
errors = {variable: error for variable, (mapping, error) in all_predictors.items()}
# Now choose the best and save that as "model"
# Sort by error
best_variable, best_error = sorted(errors.items(), key=itemgetter(1))[0]
print("The best model is based on variable {0} and has error {1:.2f}".format(best_variable, best_error))

# Choose the bset model
model = {'variable': best_variable,
         'predictor': all_predictors[best_variable][0]}
print(model)
# print(all_predictors)-->{0: ({0: 0, 1: 2}, 41), 1: ({0: 1, 1: 0}, 58), 2: ({0: 0, 1: 2}, 37), 3: ({0: 0, 1: 2}, 37)}
# print(errors)-->{0: 41, 1: 58, 2: 37, 3: 37}


# In[7]:

def predict(X_test, model):
    variable = model['variable']
    predictor = model['predictor']
    # 检测X_test每一个sample的第variable列为什么特征值
    y_predicted = np.array([predictor[int(sample[variable])] for sample in X_test])
    return y_predicted

# In[8]:

y_predicted = predict(X_test, model)
print(y_predicted)


# In[9]:

# Compute the accuracy by taking the mean of the amounts that y_predicted is equal to y_test
accuracy = np.mean(y_predicted == y_test) * 100
print("The test accuracy is {:.1f}%".format(accuracy))

# In[10]:

from sklearn.metrics import classification_report


# In[11]:

print(classification_report(y_test, y_predicted))

```
### 附：
1. numpy中的mean函数：![用法链接](https://blog.csdn.net/lilong117194/article/details/78397329)
2. assert断言：![用法链接](https://blog.csdn.net/qq_37119902/article/details/79637578)
3. shape：![用法链接](https://blog.csdn.net/by_study/article/details/67633593)
4. train_test_split参数含义：![用法链接](https://www.e-learn.cn/content/qita/780160)
5. classification_report参数含义：![用法链接](https://blog.csdn.net/genghaihua/article/details/81155200)
