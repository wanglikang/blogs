## 2019年5月22号 LSTM初体验

## 今天总结了昨天的关于keras上LSTM的实现方法，现记录如下：

+ 构建一个两层的LSTM模型：
```python
def buildLSTMModel(in_shape=(None,1),out_shape=(None,1)):
    model = Sequential()
    model.add(LSTM(32,
                          input_shape=in_shape,
                          dropout=0.1,
                          recurrent_dropout=0.5,
                          return_sequences=True,
                          implementation=1))

    #   softsign
    model.add(LSTM(64, activation='softsign',
                          dropout=0.1,
                          recurrent_dropout=0.5,
                          implementation=1))
    model.add(Dense(out_shape[1]))
    model.compile(optimizer=RMSprop(lr=1e-4), loss='mae')
    return model
```

其中，LSTM的参数return_sequences=True表示将输出作为下一层的输入，在进行LSTM堆叠的时候，除最后一层LSTM之外，，
其余的lstm都需要设置return_sequences=True.

得益于keras强大的封装，构建模型很简单，，，难的是如何构建输入数据
### LSTM类的时序数据输入格式
首先，在keras中LSTM这类循环网络都是输入3D数据（samples,timesteps,output_dim），“3D”的3的意思是有三个维度，
每个维度代表了不同的含义。
各个维度的含义分别是：
samples：代表了第几个条数据
timesteps：注意是steps。代表了每个数据中的时间序列，因为lstm输入的是序列时间，，有序列就会有一个先后顺序，那么这个timesteps就是代表了不同顺序中的不同值
output_dim：这个维度则代表了真正的数据，是每一个序列中的真正的输入数据。
下面以代码中实现的预测三角函数的结果为例子，进行讲解：

```python


#产生数据的函数，产生的数据只有一个sin函数的值
def createRowData(length):
    x = [a/10.0 for a in range(length)]
    seqY = [math.sin(b) for b in x]
    return seqY
```
生成的数据长这样：

<img src="https://raw.githubusercontent.com/wanglikang/blogs/master/articles/assets/img/img2019-5-22-1.png" width=256 height=256 />

很普通的的正弦函数数据

目标是给一段数据的y0,y1,y2,,,,,yk，让其预测接下来的数据yk+1会是什么.

下面构造lstm的输入数据：代码如下：
```python
def dataAdapter(data,k=10,feture_num = 1,out_num=1):
    len = data.shape[0]#一共有多少条数据
    sample_length = math.floor(len / k)
    x = np.zeros([sample_length,k,feture_num])
    y = np.zeros([sample_length,out_num])
    for i in range(sample_length):
        for j in range(k):
            for f in range(feture_num):
                x[i,j,f] = data[i+j,f]
        for ak in range(out_num):
            y[i,ak] = data[i+k+ak]
    return x,y
```
代码的含义下图所示，一目了然。

<img src="https://raw.githubusercontent.com/wanglikang/blogs/master/articles/assets/img/img2019-5-22-2.png"  height=256 />

然后就是训练，预测了，这部分比较简单，详细见代码：
预测的结果如下所示：训练10轮之后，预测效果如下图：

<img src="https://raw.githubusercontent.com/wanglikang/blogs/master/articles/assets/img/img2019-5-22-3.png" height=256 />

训练100轮之后，预测的效果如下：

<img src="https://raw.githubusercontent.com/wanglikang/blogs/master/articles/assets/img/img2019-5-22-4.png"  height=256 />

可以看到，随着预测轮次的增多，预测的值与真实值之间的差距越来越小。
训练500轮之后的效果：

<img src="https://raw.githubusercontent.com/wanglikang/blogs/master/articles/assets/img/img2019-5-22-5.png"  height=256 />

现在的输入的数据只有一个特征，那就是一个数字而已，，且输出的结果也只有一个结果，同样也只有一个数字而已，，如果像将输入的特征取多个改怎么办呢？
注意，，上面那个excel表格的示意图中，输入的X的每一行中数据

<img src="https://raw.githubusercontent.com/wanglikang/blogs/master/articles/assets/img/img2019-5-22-6.png"  height=256 />


都是一个数组，意味者可以将这一个数字进行扩充，变为多个特征，在本文中，就简单的取一个数字就足够了，但是在其它的场景中，则会根据实际情况，取多个特征作为输入。
下面尝试将输出维度调大一点，从1调为5试一下，
分别训练10轮，100轮，500轮，结果分别如下所示：（结果中的锯齿为每次显示了5个数据）

10轮

<img src="https://raw.githubusercontent.com/wanglikang/blogs/master/articles/assets/img/img2019-5-22-7.png"  height=256 />

100轮

<img src="https://raw.githubusercontent.com/wanglikang/blogs/master/articles/assets/img/img2019-5-22-8.png"  height=256 />


500轮

<img src="https://raw.githubusercontent.com/wanglikang/blogs/master/articles/assets/img/img2019-5-22-9.png"  height=256 />

此时的输入数据和对应的输入结果数据为：

<img src="https://raw.githubusercontent.com/wanglikang/blogs/master/articles/assets/img/img2019-5-22-10.png" width=256 height=256 />
