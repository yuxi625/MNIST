import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
from CNN import CNN
#############数据加载####################


train_data = dataset.MNIST(
    root = "./mnist",
    train = True,
    transform = transforms.ToTensor(),
    download = True
)

test_data = dataset.MNIST(
    root = "./mnist",
    train = False,
    transform = transforms.ToTensor(),
    download = True
)

# print(train_data)
# print(test_data)


##################分批加载#############
train_loader = data_utils.DataLoader(
    dataset = train_data,
    batch_size = 64,
    shuffle = True,
)
test_loader = data_utils.DataLoader(
    dataset = test_data,
    batch_size = 64,
    shuffle = True,
)

# print(train_loader)
# print(test_loader)
cnn = CNN()
cnn = cnn.cuda()


####################损失函数################
loss_func = torch.nn.CrossEntropyLoss()
###################优化函数################
optimizer = torch.optim.Adam(cnn.parameters(),lr=0.01)


###############训练过程####################
epoch =10
for e in range(epoch):
    # 训练集训练
    for index,(images,labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()
        # 前向传播
        outputs = cnn(images)
        # 传入输出层节点和真实标签来计算损失函数
        loss = loss_func(outputs,labels)
        # 先清空梯度
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        optimizer.step()
        # print("当前为第{}轮，当前批次为{}/{},loss为{}".format(e+1,index+1,len(train_data)//64,loss.item()))

    # 测试集验证
    loss_test = 0
    rightValue = 0
    for index,(images,labels) in enumerate(test_loader):
        images = images.cuda()
        labels = labels.cuda()
        outputs = cnn(images)
        loss_test += loss_func(outputs, labels)
        _, pred = outputs.max(1)
        rightValue += (pred==labels).sum().item()
        # print("当前为第{}轮测试集验证,当前批次为{}/{},loss为{},准确率是{}".format(e + 1, index + 1, len(test_data) // 64, loss_test, (pred==labels).sum().item()/64))
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$${}".format(rightValue/len(test_data)))

torch.save(cnn,"./model/mnist_model.pkl")










