# 用户新增预测挑战赛任务一：Baseline
**赛题名称**：用户新增预测挑战赛
**赛题类型**：数据挖掘、二分类
**赛题链接**👇：
https://challenge.xfyun.cn/topic/info?type=subscriber-addition-prediction&ch=ymfk4uU

## 赛题背景
讯飞开放平台针对不同行业、不同场景提供相应的AI能力和解决方案，赋能开发者的产品和应用，帮助开发者通过AI解决相关实际问题，实现让产品能听会说、能看会认、能理解会思考。

用户新增预测是分析用户使用场景以及预测用户增长情况的关键步骤，有助于进行后续产品和应用的迭代升级。

## 赛事任务
本次大赛提供了讯飞开放平台海量的应用数据作为训练样本，参赛选手需要基于提供的样本构建模型，预测用户的新增情况。

## 赛题数据集
赛题数据由约62万条训练集、20万条测试集数据组成，共包含13个字段。

其中uuid为样本唯一标识，eid为访问行为ID，udmap为行为属性，其中的key1到key9表示不同的行为属性，如项目名、项目id等相关字段，common_ts为应用访问记录发生时间（毫秒时间戳），其余字段x1至x8为用户相关的属性，为匿名处理字段。target字段为预测目标，即是否为新增用户。

## 评价指标
本次竞赛的评价标准采用f1_score，分数越高，效果越好。

## 我的思路和笔记
首先我们将一些必备的库函数导入并把比赛的数据读入

```python
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier   #导入分类决策树

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
train_data.head()
```

我们会发现所有的字段几乎都是（整型）数字，除了```udmap``` 对应的元素数据类型是字典。为了将这个内容转换为数字，我们可以先把 ```udmap``` 特征进行了预处理，将其转换为一个长度为 $9$ 的向量，$x_{i}$ 的值代表的是 **keyi** 键对应的价值。

**Tips：之后所有的处理都是训练集与测试集一起处理的**

```python
def udmap_onethot(d):
    v = np.zeros(9)
    if d == 'unknown':
        return v
    
    d = eval(d)
    for i in range(1, 10):
        if 'key' + str(i) in d:
            v[i-1] = d['key' + str(i)]

    return v
```

我们可以看到，有些 ```udmap```的内容为 “**unknown**”，所以我们写一个分支并输出 $[0, 0, 0, 0, 0, 0, 0, 0, 0]$。之后我们把这个函数应用一下：

```python
train_udmap_df = pd.DataFrame(np.vstack(train_data['udmap'].apply(udmap_onethot))) 
test_udmap_df = pd.DataFrame(np.vstack(test_data['udmap'].apply(udmap_onethot))) 
```

我来一步步解释等号右边的代码

1. ```train_data['udmap']```：从 *train_data* 数据框中选择名为 *udmap* 的列。

2. ```.apply(udmap_onethot)```：对 *train_data['udmap']* 列中的每个元素应用 *udmap_onethot* 函数。

3. ```np.vstack()```：将 *udmap_onethot* 函数应用后的结果垂直堆叠成一个数组。

4. ```pd.DataFrame(...)```：将垂直堆叠后的数组转换为一个新的数据框 *train_udmap_df*。

之后我们把这些堆叠之后的数组横向拼接起来

```python
train_udmap_df.columns = ['key' + str(i) for i in range(1, 10)]
test_udmap_df.columns = ['key' + str(i) for i in range(1, 10)]
train_data = pd.concat([train_data, train_udmap_df], axis=1)
test_data = pd.concat([test_data, test_udmap_df], axis=1)
```

第一行好像用了魔法的方法命名了 **key1~key9**

```python
train_data['eid_freq'] = train_data['eid'].map(train_data['eid'].value_counts())
test_data['eid_freq'] = test_data['eid'].map(train_data['eid'].value_counts())

train_data['eid_mean'] = train_data['eid'].map(train_data.groupby('eid')['target'].mean())
test_data['eid_mean'] = test_data['eid'].map(train_data.groupby('eid')['target'].mean())
```

我们将数据用出现频率作为一个特征进行标准化。

```python
train_data['udmap_isunknown'] = (train_data['udmap'] == 'unknown').astype(int)
test_data['udmap_isunknown'] = (test_data['udmap'] == 'unknown').astype(int)
train_data['common_ts_hour'] = train_data['common_ts'].dt.hour
test_data['common_ts_hour'] = test_data['common_ts'].dt.hour
```

创建一个新的列 ```udmap_isunknown```，该列的值为 **0** 或 **1**，表示```udmap```列中对应行的取值是否为 **'unknown'** 。如果 ```udmap``` 列中对应行的取值为*'unknown'**，则* *```udmap_isunknown```* *列的值为 \*1**，否则为 **0**。

将```train_data```数据集中```common_ts```列的时间类型数据转换为小时数，并将结果保存到新的列```common_ts_hour```中。

```python
clf = DecisionTreeClassifier()
clf.fit(
    train_data.drop(['udmap', 'common_ts', 'uuid', 'target'], axis=1),
    train_data['target']
)
```

此段代码功能为加载决策树模型进行训练(直接使用sklearn中导入的包进行模型建立)

其中我们投喂的数据不包括```udmap```, ```common_ts```, ```uuid```, ```target```。我们的目标是```target```

```python
pd.DataFrame({
    'uuid': test_data['uuid'],
    'target': clf.predict(test_data.drop(['udmap', 'common_ts', 'uuid'], axis=1))
}).to_csv('submit.csv', index=None)
result_df.to_csv('submit.csv', index=None)
```

创建一个 *DataFrame* 来存储预测结果，其中包括两列：```uuid``` 和 ```target```

```uuid``` 列来自测试数据集中的 ```uuid```列，```target``` 列将用来存储模型的预测结果

最后将结果 *DataFrame* 保存为一个 *CSV* 文件，文件名为 *'submit.csv'*
