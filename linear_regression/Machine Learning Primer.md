## Machine Learning Primer
v2.0 by [KzXuan](https://github.com/kzxuan), NUSTM, 2020.04.30

**本材料面向NUSTM研究组的新生，帮助你们系统性地上手机器学习，更快地掌握相关知识并达到组内的基本代码水平要求。** 我不会在这里提供相关的知识，但是梳理了学习的流程和材料，希望你们可以按序全部完成。

</br>

#### 需要预备的基础知识包括：

* Python编程，熟练掌握语法、类操作等
* Numpy包，了解基本的向量、矩阵操作
* 数学知识，常规求导、线性代数、概率论等

</br>

#### 主要的学习材料包括：

* NJUST Machine Learning英文课程课件 (by Rui Xia)
* Coursera Machine Learning网络课程 (by Andrew Ng)
* 机器学习 (by 周志华)
* 辅助资料：[机器学习学习笔记 (by Vay-keen)](<https://github.com/Vay-keen/Machine-learning-learning-notes>)

</br>

### 0. Machine Learning Introduction

* 阅读 Machine Learning Introduction (by Rui Xia)。
* 学习 [Coursera Introduction](<https://www.coursera.org/learn/machine-learning/home/week/1>) 部分，并完成课程测试。
* 阅读 统计学习方法-第一章-统计学习方法概论 (by 李航)。
* 阅读 机器学习-第一章-绪论 (by 周志华)。

> **思考**
> 1. 什么是机器学习，机器学习在做什么？
> 2. 什么是分类，什么是回归？
> 3. 有监督和无监督分类的区别是什么？

</br>

### 1. Linear Regression

**理解线性回归以及梯度下降法(GD)。**

* 阅读 Linear Regression (by Rui Xia)，并**推导其中的所有公式**，然后**独立**完成课后作业。
* 学习 [Coursera Linear Regression with One Variable & Linear Algebra Review](<https://www.coursera.org/learn/machine-learning/home/week/1>) 部分，并完成课程测试。
* 学习 [Coursera Linear Regression with Multiple Variables](<https://www.coursera.org/learn/machine-learning/home/week/2>) 部分，并完成课程测试。

> **思考**
> 1. 在数据中添加一列1有什么作用？否则的话是否需要添加什么参数？
> 2. 如何初始化权重？全0或全1权重是否可以？
> 3. 模型预测和更新时如何使用numpy矩阵进行操作来代替复杂的循环？
> 4. 为什么要进行归一化，使用不同的归一化方法是否会有所差异？如何对测试数据进行归一化？
> 5. 调节学习率(learning_rate)和迭代次数(iter_times)会对结果产生什么样的影响？
> 6. 除了迭代次数，还有什么样的条件可以作为训练终止的判断依据？
> 7. [扩展] 动态作图是如何实现的？

</br>

### 2. Logistic Regression

**理解逻辑回归的本质是分类，理解随机梯度下降法(SGD)和牛顿法。**

* 阅读 Logistic Regression (by Rui Xia)，并**推导其中的所有公式**，然后**独立**完成课后作业。

> **思考**
> 1. 如何在代码中体现SGD中的随机性？不同的随机方法可能带来什么样的差异？
> 2. 为什么在SGD中引入学习率衰减系数(learning_rate_decay)，其作用是什么？
> 3. 为什么牛顿法收敛得非常快？
> 4. [扩展] 如何求出并画出逻辑回归的分类线？

</br>

### 3. Softmax Regression

**理解Softmax回归和逻辑回归的相同和不同之处。**

* 阅读 Softmax Regression (by Rui Xia)，并**推导其中的所有公式**，然后**独立**完成课后作业。
* 学习 [Coursera Logistic Regression](<https://www.coursera.org/learn/machine-learning/home/week/3>) 部分，并完成课程测试。
* 学习 [Coursera Regularization](<https://www.coursera.org/learn/machine-learning/home/week/3>) 部分，并完成课程测试。
* 阅读 统计学习方法-第六章-逻辑斯谛回归与最大熵模型 (by 李航)。

> **思考**
> 1. 什么是one-hot？为什么标签要被表示为one-hot形式？
> 2. 最大似然估计/交叉熵损失的计算与逻辑回归有何不同？
> 3. 引入画图间隔步数(display_step)参数有什么用处？
> 4. 模型代码能否如何不同类别数的分类数据？二分类和三分类数据下，所设置的参数有什么区别吗？
> 5. [扩展] 除了一次训练所有样本(GD)和一次训练一个样本(SGD)，还有没有什么更好的训练方式？
> 6. [扩展] 为什么要通过遍历点的方式绘制Softmax回归的分类边界？还有没有什么简单的方法可以获得分类边界？

</br>

### 4. Perceptron

**理解感知机的损失计算策略和更新策略。**

* 阅读 Perceptron (by Rui Xia)，并**推导其中的所有公式**，然后**独立**完成课后作业。
* 阅读 统计学习方法-第二章-感知机 (by 李航)。

> **思考**
> 1. 如何选取更新点？
> 2. 什么是pocket算法？为什么要在感知机中引入pocket算法？
> 3. 不使用pocket算法感知机能否趋于稳定？使用与不使用对其它参数的设置有什么影响？
> 4. 多分类和标准二分类感知机在实现上有什么异同？多分类感知机和Softmax回归在实现上又有什么异同？

</br>

### 5. Linear Models Review

**回顾线性模型，理解它们在模型假设、学习和决策三个步骤上的相似和不同之处。**

* 阅读 Linear Models Review (by Rui Xia)
* 阅读 机器学习-第三章-线性模型 (by 周志华)

</br>

### 6. Artificial Neural Network

**理解神经网络模型的前向传播和反向更新过程。**

* 阅读 Artificial Neural Network (by Rui Xia)，并**推导其中的所有公式**，然后**独立**完成课后作业。
* 学习 [Coursera Neural Networks: Representation](<https://www.coursera.org/learn/machine-learning/home/week/4>) 部分，并完成课程测试。
* 学习 [Coursera Neural Networks: Learning](<https://www.coursera.org/learn/machine-learning/home/week/5>) 部分，并完成课程测试。
* 阅读 机器学习-第五章-神经网络 (by 周志华)。

> **思考**
> 1. 神经网络网络模型的参数初始化和之前的模型有什么不同吗？
> 2. 不同激活函数的公式、曲线和导数分别是什么样的？各有什么作用或特点？
> 3. 为什么将神经网络的输出层的激活函数固定为Softmax？为什么使用交叉熵作为损失函数？如何计算此时的反向梯度？
> 4. [扩展] 如何设计一个可任意更改隐藏层层数、节点数和激活函数的模型代码？
> 5. [扩展] 怎样将不同功能的模块拆分开来，使代码更简洁、功能更清晰？能否依此对完成的代码进行重构？

</br>

### 7. Naive Bayes

**理解多项式模型和多变量伯努利的区别。**

* 阅读 Naive Bayes (by Rui Xia)，并**推导其中的所有公式**，然后**独立**完成课后作业。
* 阅读 统计学习方法-第四章-朴素贝叶斯法 (by 李航)。
* 阅读 机器学习-第七章-贝叶斯分类器 (by 周志华)。

> **思考**
> 1. 是否有必要去除停用词？去除停用词会对模型结果产生怎样的影响？
> 2. 为什么用log值加和的计算方式取代概率乘积？
> 3. 为什么Bernoulli模型的计算速度较慢？有什么改进的方法吗？
> 4. 在Bernoulli模型中参数alpha起到什么作用？不同的alpha值所对应的平滑方式是否有所区别？
> 5. 如何分析两个模型所得准确率的差异？

</br>

### 知识扩展

* 掌握工具 [LibNB (by Rui Xia)](<https://github.com/rxiacn/LibNB>)，和自己的代码进行对比
* 掌握工具 [PyTC (by Rui Xia)](<https://github.com/rxiacn/PyTC>)，熟悉其中的函数及使用
* 掌握示例代码中和画图相关的部分，熟悉Matplotlib的用法
* 完成 [Coursera Machine Learning](<https://www.coursera.org/learn/machine-learning/home/welcome>) 课程的剩余部分
* 阅读 统计学习方法 (by 李航) 的剩余部分
* 阅读 机器学习 (by 周志华) 第二、四、六、十一、十四章