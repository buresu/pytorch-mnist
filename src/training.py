import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

from model.net import Net


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

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)

    epochs = 10

    # training
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statostics
            running_loss += loss.item()
            if i % 100 == 99:
                print('[{:d}, {:5d}] loss: {:.3f}'.format(epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Training Finished')

    torch.save(model.state_dict(), './model/test.model')

    # test
    correct = 0
    total = 0
    with torch.no_grad():
        for (images, labels) in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy: {:.2f} %'.format(100 * float(correct/total)))
