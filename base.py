import argparse
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from opacus import PrivacyEngine

from models import CNNMnist, CNNCifar

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=1000)

parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='Adam weight decay')
parser.add_argument('--seed', type=int, default=1, help='random seed')

parser.add_argument('--activation', type=str, default='relu', help='activation function')
parser.add_argument('--scale', type=float, default=2.0, help='scale for tempered sigmoid')
parser.add_argument('--temp', type=float, default=2.0, help='temperature for tempered sigmoid')
parser.add_argument('--offset', type=float, default=1.0, help='offset for tempered sigmoid')

parser.add_argument('--disable_dp', action='store_true', help='disable differential privacy')
parser.add_argument('--epsilon', type=float, default=2.93, help='epsilon for (epsilon, delta)-DP')
parser.add_argument('--delta', type=float, default=1e-5, help='delta for (epsilon, delta)-DP')
parser.add_argument('--max_norm', type=float, default=0.1, help='clip per-sample gradients')
parser.add_argument('--sigma', type=float, default=1.0, help='noise multiplier')
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print('Device: ', device, flush=True)

if args.dataset == 'mnist' or args.dataset == 'fmnist':
    model = CNNMnist(args=args)
elif args.dataset == 'cifar':
    model = CNNCifar(args=args)
else:
    exit('Error: unrecognized dataset')
model.to(device)

data_dir = './data'
if args.dataset == 'mnist':
    apply_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                transform=apply_transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                transform=apply_transform)
elif args.dataset == 'fmnist':
    apply_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                        transform=apply_transform)
    test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                        transform=apply_transform)
elif args.dataset == 'cifar':
    apply_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                    transform=apply_transform)
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                    transform=apply_transform)
else:
    exit('Error: unrecognized dataset')

if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum)
elif args.optimizer == 'adam':
    args.lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay)

trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
criterion = torch.nn.NLLLoss().to(device)
epoch_loss = []

if not args.disable_dp:
    print('Differential Privacy is enabled', flush=True)
    privacy_engine = PrivacyEngine()
    model, optimizer, trainloader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=trainloader,
        epochs=args.epochs,
        target_epsilon=args.epsilon,
        target_delta=args.delta,
        max_grad_norm=args.max_norm,
    )
    print(f"Using sigma={optimizer.noise_multiplier} and C={args.max_norm}", flush=True)
else:
    print('Differential Privacy is disabled', flush=True)

def train(epoch, model, trainloader, optimizer, criterion, device):
    model.train()
    for batch_idx, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch+1, batch_idx * len(images), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.item()), flush=True)

def test(model, testloader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_loss += criterion(outputs, labels).item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
    
    test_loss /= len(testloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset), flush=True))

for epoch in range(args.epochs):
    train(epoch, model, trainloader, optimizer, criterion, device)
    test(model, testloader, device)