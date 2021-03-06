import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import numpy as np

from model.net import Net


def evaluate(loader, model, epoch):

    correct = 0
    total = 0
    accuracy = 0.0
    with torch.no_grad():
        for (images, labels) in loader:

            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = float(correct/total)
        print('Epoch: {:d} Accuracy: {:.2f} %'.format(epoch, 100 * accuracy))
        torch.save(model.state_dict(), './model/test_{:d}epoch_{:.2f}%.model'.format(epoch, 100 * accuracy))

    return accuracy

if __name__ == "__main__":
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    trainset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=100,
        shuffle=True,
        num_workers=2)

    testset = torchvision.datasets.MNIST(
        root='data',
        train=False,
        download=True,
        transform=transform)

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=100,
        shuffle=False,
        num_workers=2
    )

    classes = tuple(np.linspace(0, 9, 10, dtype=np.uint8))

    # model
    model = Net()

    # teansorboad
    writer = SummaryWriter()
    x, y = next(iter(trainloader))
    writer.add_graph(model, x)

    if torch.cuda.is_available():
        model.cuda()

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.99)

    epochs = 10
    every_eval = 10
    min_eval = 10

    # training
    num_iteration = 0
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):

            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            num_iteration += 1

            # print statostics
            running_loss += loss.item()
            if i % 100 == 99:
                print('[{:d}, {:5d}] loss: {:.3f}'.format(epoch + 1, i + 1, running_loss / 100))
                writer.add_scalar("training/loss", running_loss, num_iteration)
                running_loss = 0.0

        writer.add_scalar("training/learning_rate", float(optimizer.param_groups[0]['lr']), epoch)

        scheduler.step()

        if (epoch + 1) % every_eval == 0 and (epoch + 1) >= min_eval:
            accuracy = evaluate(testloader, model, epoch + 1)
            writer.add_scalar("training/accuracy", accuracy, epoch)

    print('Training Finished')

    evaluate(testloader, model, epochs)

    writer.close()
