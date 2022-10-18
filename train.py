import torchvision
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import nn
from model import nn_Test

train_data = torchvision.datasets.CIFAR10(root="CIFAR10", train=True,
    transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(root="CIFAR10", train=False,
    transform=torchvision.transforms.ToTensor(), download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

print("the length of train set is {}".format(train_data_size))
print("the length of test set is {}".format(test_data_size))

train_dataloader =DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

writer = SummaryWriter("logs")

# model
nn_test = nn_Test() # from model.py

loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-2
optimizer = torch.optim.SGD(nn_test.parameters(), lr=learning_rate)

total_train_step = 0
epoch = 10

for i in range(epoch):
    print("----------第{}轮训练开始----------".format(i+1))

    for data in train_dataloader:
        imgs, targets = data
        outputs = nn_test(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数:{}, Loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train loss", loss.item(), total_train_step)

    total_rn = 0
    total_test_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = nn_test(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            right_num = (outputs.argmax(1) == targets).sum() # 正确的个数
            total_rn += right_num

    accuracy = total_rn/test_data_size
    print("整体测试集上的Loss:{}",format(total_test_loss))
    print("整体测试集上的accuracy:{}".format(accuracy))
    writer.add_scalar("test_loss", total_test_loss, i)
    writer.add_scalar("test_accuracy", accuracy, i)


    torch.save(nn_test, "model_train/nn_test_{}.pth".format(i))
    print("模型已保存")

writer.close()
