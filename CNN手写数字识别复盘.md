# CNN复盘

#### 1、加载数据集

​		目的：从开源的数据集中下载数据（MNIST），并进行数据预处理

​		用到的库：

​			--torch  这个相当于总的库 直接加载进来，但是其实只是用其中的特定的包

​			--torchvision.transforms 这个是用来进行数据预处理的，包括（是否随机，平均化、旋转图片、图片张量化等）

​			--torchvision.datasets 这个是相当于负责处理数据集的，单纯提供数据的

​			--torch.utils.data.DataLoader 这个是加载数据的，用来最终交给网络训练处理的

![img](https://github.com/xiesixiong/CNN_LexNet5-/blob/main/image.png)

​		代码展示：

```python
train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.Compose([transforms.ToTensor(),
                                                          transforms.Normalize(mean=0.5,std=0.5)]),
                            download=True)
test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.Compose([transforms.ToTensor(),
                                                        transforms.Normalize(mean=0.5,std=0.5)]),
                           download=True)

train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)

#划分数据集
index =range(len(test_dataset))
index_val = index[:4000]
index_test = index[4000:]

#使用采样器在数据集中进行有选择的采样
sampler_val = torch.utils.data.sampler.SubsetRandomSampler(index_val)
sampler_test =  torch.utils.data.sampler.SubsetRandomSampler(index_test)

#利用dataloader将采样器中的数据加载构造数据集
val_loader = DataLoader(dataset= test_dataset,batch_size=batch_size,shuffle =False,sampler = sampler_val,num_workers=num_workers)
test_loader= DataLoader(dataset= test_dataset,batch_size=batch_size,shuffle =False,sampler = sampler_test,num_workers=num_workers)

```

​	数据阶段：

​		分为：下载原始数据、数据预处理（transformer）、划分数据集、数据加载（dataloader）四大模块



#### 2、数据可视化

​		图片本质上在计算机中就是单纯的像素值，可能各个维度负责的颜色不一样，但是本质上就是单纯的数字，所以给定一个二维数组，就可以生成一张图片 ，只是图片有没有真实的物理意义就另说了，所以，当所分配的数字满足一定规律时，就展示给我们一副有意义的画面了。



​		代码展示：

```python
fig = plt.figure()
for i in range(12):
    plt.subplot(3, 4, i+1)
    plt.tight_layout()
    plt.imshow(train_dataset.train_data[i], cmap='gray', interpolation='none')
    plt.title("Labels: {}".format(train_dataset.train_labels[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()
```

​		效果展示：

<img src="C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20221220130648506.png" alt="image-20221220130648506" style="zoom: 50%;" />

#### 3、搭建CNN网络架构



​		目的：通过建立卷积网络，去提取每个数字图像的特征，然后连接全连接层，将提取出的特征融合，作为分类的依据。

​		用到的库：

​			torch.nn ：这个是pytorch的所提供的负责搭建模型时集成的API(包括了卷积、全连接、池化、激活等常用操作)

​			

​		**模型架构：**

​			输入层In（1 * 28 * 28）



​			卷积层C1（filter：10@5*5）

​			激活层S1（ReLu）

​			池化层P1（MaxPool）



​			卷积层C2（filter：20@5*5）

​			激活层S2（ReLu）

​			池化层P2（MaxPool）

​	

​			

​			全连接层Fc1（10）



效果图展示：

![image-20221220122530206](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20221220122530206.png)

代码展示：

```python
class CNN(nn.Module):
    def __init__(self):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        super(CNN, self).__init__()
        # 卷积层conv1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU())
        # 池化层pool1
        self.pool1 = nn.MaxPool2d(2)
        # 卷积层conv2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU())
        # 卷积层conv3
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        # 池化层pool2
        self.pool2 = nn.MaxPool2d(2)
        # 全连接层fc(输出层)
        self.fc = nn.Linear(5 * 5 * 64, 10)

```

​			

#### 4、损失函数和优化器

##### 损失函数使用**交叉熵损失**

​		交叉熵损失函数代码：

```python
criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失
```

​		交叉熵函数数学公式：

![image-20221220123748802](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20221220123748802.png)

​		为什么选择交叉熵函数：

​				交叉熵能够衡量同一个随机变量中的两个不同概率分布的差异程度，在深度学习中就表示为真实概率分布与预测概率分布之间的差异。



##### 参数优化使用**随机梯度下降**

​		随机梯度优化器代码：

```python
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)  # lr学习率，momentum冲量
```

​		算法原理：

<img src="https://pic2.zhimg.com/80/v2-257eb40d37bb8e483a9699e65516a3ed_720w.webp" alt="img" style="zoom:67%;" />

​				

​					通过计算梯度，不断找出使得计算出来的损失函数下降最快的方向，使得损失函数的值快速减小



#### 5、开始训练

​		目的：利用训练集训练模型参数，找到一套能够区分各个图片的神经网络结构

​		步骤：

​			1）从训练集中获取数据（图像+标签）

​			2）将图像作为输入传入模型中进行计算

​			3）利用损失函数计算模型的输出与标签的差距

​			4）计算梯度，反向传播更新网络参数

​			5）完成一次epoch，可以进行多次epoch训练

​			6）最终保存模型

```python
    for epoch in range(1):
        print("这是第{}轮epoch".format(epoch))
        for i,(image,label) in enumerate(train_dataloader):
            image = image
            label = label
            my_model.train()
            output = my_model(image)
            loss = loss_function(output, label)
            optim.zero_grad()
            loss.backward()
            optim.step()
            if i%10 == 0:
                print("训练次数{}，损失率{}".format(i+1,loss.item()))
                step+=1
                writer.add_scalar("loss",loss,step)
                
    torch.save(my_model,"model.pth")
```

​			

#### 6、测试集验证

​			目的：检验此次模型训练的效果，即验证该套网络参数的准确度

​			步骤：

​				1）从测试集中获取数据（图像+标签）

​				2）将图像作为输入传入模型中进行计算，得到预测的标签

​				3）和原有的标签进行比对，记录预测正确的个数

​				4）计算模型的准确率

```python
if __name__ == '__main__':
    #1.将数据转为标准的数据格式
    test_data = data_processing.standard_data("../mnist/DataImages-Test")
    #2.加载数据集
    test_data = DataLoader(test_data,batch_size=1,shuffle=True)
    #3.加载模型
    model = torch.load("model.pth")
    #4.对测试集挨个用模型预测
    correct = 0
    for i,(image,label) in enumerate(test_data):
        #4.1由于在标准化模型的过程中是展开的，利用one-hot转码的时候需要吧矩阵转化回去
        label = label.view(config.get_number_of_captcha(),-1)# torch.Size([4, 62])

        #进行one-hot转化为文本
        label = one_hot.vec_to_text(label)

        #4.2将图片矩阵进行预测
        output = model(image)
        output = output.view(config.get_number_of_captcha(), -1)

        output = one_hot.vec_to_text(output)
        # print("标签值为{}，预测值为{}".format(label, output))
        # print("label:",label)
        # print("output:",output)
        if output==label:
            correct+=1

    print("正确率为",correct/test_data.__len__())

```

​				模型准确率展示：

<img src="C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20221220132951619.png" alt="image-20221220132951619" style="zoom: 67%;" />

![image-20221220133028118](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20221220133028118.png)

​				从图中我们不难看出，模型在第8个epoch已经开始收敛了，最终的准确率能达到98.7%，模型预测效果较好。

#### 7、总结复盘：

​		本次实验，利用了CNN卷积神经网络在mnist数据集上进行手写数字识别的分类训练，属于图像处理领域一个入门实验。

通过完成本次实验，让我对图像识别整体的流程有了一个更深入的认识，现总结如下:

​		<1>数据集处理：将原本的图像处理经过各种操作，处理成神经网络可以接受的格式

​		<2>搭建模型：本次用到的是卷积神经网络，不同于传统的全链接网络，卷积网络参数更少，提取图像特征效果更显著，

​									如何构建一套网络架构决定了模型最终的效果好坏，也就是图像分类的准确率，但是目前我还不能自己									搭建网络的架构，对每个网络层的本质还没有更加深刻的理解，这是未来学习的重点。

​		<3>模型训练：这个过程的目的是让网络不断找到最合适的神经元之间的参数，起到关键的两部分分别是：损失函数、优									化器。通过计算出模型预测的结果与真实标签的差距作为损失，利用优化器来方向对网络参数进行优									化，从而不断调整模型。

​		<4>模型验证：即利用测试集来验证训练好的模型，计算模型的预测准确度，从而来评价模型的好坏。		

​		深度学习虽然集成的库有很多，但是各个算法原理思想还是需要自己实践领会，不能单纯简单的知道如何调用这些函数，更要知道为什么调用这些函数，以及函数的具体含义是什么，其中的注意事项是什么，真正做到“知其然且知其所以然”。深度学习领域优秀的算法思想还有许多，需要我们怀着谦虚的心态继续学习，希望有朝一日能够搭建出属于自己的网络。
