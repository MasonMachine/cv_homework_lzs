# 手写数字识别
import torch
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from model import  *
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# 参数设置
num_epochs = 10
batch_size = 64
learning_rate = 0.01

# 将数据处理成Variable, 如果有GPU, 可以转成cuda形式
def get_variable(x):
    x = Variable(x)
    return x.cuda() if torch.cuda.is_available() else x

# 对数据进行载入及有相应变换,将Compose看成一种容器，他能对多种数据变换进行组合
# 传入的参数是一个列表，列表中的元素就是对载入的数据进行的各种变换操作(只有一个颜色通道)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, ], std=[0.5, ])])
# 获取MNIST训练集和测试集
data_train = datasets.MNIST(root='data/', transform=transform, train=True, download=True)
data_test = datasets.MNIST(root='data/', transform=transform, train=False)
# 数据装载
data_loader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
data_loader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size=batch_size, shuffle=True)

# 对模型进行训练和参数优化
cnn_model = CNN_Model()
# 将所有的模型参数移动到GPU上
if torch.cuda.is_available():
    cnn_model = cnn_model.cuda()

loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=learning_rate)
if torch.cuda.is_available():
    loss_func = loss_func.cuda()

for epoch in range(num_epochs):
    running_loss = 0.0
    running_correct = 0.0
    print("Epoch  {}/{}".format(epoch, num_epochs))
    cnn_model.train()
    for data in data_loader_train:
        X_train, y_train = data
        X_train, y_train = get_variable(X_train), get_variable(y_train)
        outputs = cnn_model(X_train)
        _, pred = torch.max(outputs.data, 1)
        optimizer.zero_grad()
        loss = loss_func(outputs, y_train)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_correct += torch.sum(pred == y_train.data)
    testing_correct = 0.0
    for data in data_loader_test:
        X_test, y_test = data
        X_test, y_test = get_variable(X_test), get_variable(y_test)
        outputs = cnn_model(X_test)
        _, pred = torch.max(outputs, 1)
        testing_correct += torch.sum(pred == y_test.data)
    print("Loss is :{:.4f},Train Accuracy is:{:.4f}%,Test Accuracy is:{:.4f}%".format(
        running_loss / len(data_train), 100 * running_correct / len(data_train),
        100 * testing_correct / len(data_test)))
# 保存模型
torch.save(cnn_model, 'models/cnn_model.pt')
print("模型已保存")
