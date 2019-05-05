## 数据结构与算法  课程大作业：WFC算法实现

### 摘要

本项目是2019年春季数据结构与算法课程大作业，由孟凡强，孙亮，蔡斌共同完成。我们用python实现了波函数塌缩算法(WaveFunctionCollapse), 并试图进行一些运用。本文将主要包括：

- WFC算法简介
- 需求分析与算法实现
- 示例
- ......



### WFC算法简介

WaveFunctionCollapse 算法由Maxim Gumin等人于2016年提出，最早呈现于一个GitHub项目[WFC](<https://github.com/mxgmn/WaveFunctionCollapse>).  其算法的构想借鉴于量子力学中的波函数塌缩概念，用于生成与输入位图局部相似的位图。算法并未采用机器学习的方式，而主要是一种非回溯的贪心搜索算法。
随着研究的深入[^1]，人们
[^1]: <https://adamsmith.as/papers/wfc_is_constraint_solving_in_the_wild.pdf>





### 构想(雾

- #### 数据结构

  - 格点：记录该点的状态空间和(香农)熵。状态空间为可取状态及其权重的字典 { state : weight }。香农熵具体定义参见[Entropy(Information theory)](<https://en.wikipedia.org/wiki/Entropy_(information_theory)>)

  $$
  S=-\sum_i \left(p_i \ln p_i\right)=-\sum_i \frac{w_i}{\sum w_i} \ln{\frac{w_i}{\sum w_i}}
  $$

  - (整体)波函数：包含格点的矩阵，为了方便单独记录其大小(size)。

- #### 核心思想

  寻找熵最小的点  ->  根据权重随机塌缩  ->  在塌缩点附近传播影响  ->  循环

- #### 目前的问题

  - 影响(或者约束)的代码描述，这一块的内容最为复杂，也是程序核心所在
  - 优化塌缩过程，比如在寻找熵最小的点的过程中，直接跳过已经塌缩的点，尽量不做全局搜索
  - 与图形的接口，需要与图片处理相连接
  - ......
