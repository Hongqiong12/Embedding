# Embedding
## 一、通过修改原始WORD2VEC算法，实现ARIBNB中将order商品作为全局上下文。

可以通过运行embedding下的AribnbOrdPredict实现。运行代码中给定例子可以看出，对于序列为：
9744_81732,2679_372273,2679_411245,2679_593370,4833_-1,4833_0,9434_19306,9434_7623,9434_89486,9435_12310,9435_26909,9435_422036,-1

的seesion，尾部字符为-1，说明该序列的倒数第二个商品为下单的商品，因此当前面每个词作为中心词的时候，其窗口词中都应该含有9435_422036，即下标为11的商品

----中心词为:9744_81732,随机数为：b=2 则0 的窗口大小为3的词下标为：1 ,2 ,3 ,11 ,  
----中心词为:2679_372273,随机数为：b=4 则1 的窗口大小为1的词下标为：0 ,2 ,11 ,  
----中心词为:2679_411245,随机数为：b=1 则2 的窗口大小为4的词下标为：0 ,1 ,3 ,4 ,5 ,6 ,11 ,  
----中心词为:2679_593370,随机数为：b=0 则3 的窗口大小为5的词下标为：0 ,1 ,2 ,4 ,5 ,6 ,7 ,8 ,11 ,  
----中心词为:4833_-1,随机数为：b=1 则4 的窗口大小为4的词下标为：0 ,1 ,2 ,3 ,5 ,6 ,7 ,8 ,11 ,  
----中心词为:4833_0,随机数为：b=2 则5 的窗口大小为3的词下标为：2 ,3 ,4 ,6 ,7 ,8 ,11 ,  
----中心词为:9434_19306,随机数为：b=1 则6 的窗口大小为4的词下标为：2 ,3 ,4 ,5 ,7 ,8 ,9 ,10 ,11 ,  
----中心词为:9434_7623,随机数为：b=0 则7 的窗口大小为5的词下标为：2 ,3 ,4 ,5 ,6 ,8 ,9 ,10 ,11 ,  
----中心词为:9434_89486,随机数为：b=3 则8 的窗口大小为2的词下标为：6 ,7 ,9 ,10 ,11 ,  
----中心词为:9435_12310,随机数为：b=3 则9 的窗口大小为2的词下标为：7 ,8 ,10 ,11 ,  
----中心词为:9435_26909,随机数为：b=3 则10 的窗口大小为2的词下标为：8 ,9 ,11  

其中有一个要注意的点，在WORD2VEC中，当我们设置窗口的大小是5的时候，实际上更新的过程中窗口大小并不是真正的5，而是5-（随机初始化的一个1-5之间的值）。

论文见：https://www.kdd.org/kdd2018/accepted-papers/view/real-time-personalization-using-embeddings-for-search-ranking-at-airbnb
