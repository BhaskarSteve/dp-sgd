import argparse
import torch
from opacus import PrivacyEngine
from torch.utils.data import TensorDataset, DataLoader

from models import ScatterLinear, ScatterCNN
from utils import get_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=64)

parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='Adam weight decay')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--model', type=str, default='linear', help='linear or cnn')

parser.add_argument('--disable_dp', action='store_true', help='disable differential privacy')
parser.add_argument('--epsilon', type=float, default=2.93, help='epsilon for (epsilon, delta)-DP')
parser.add_argument('--delta', type=float, default=1e-5, help='delta for (epsilon, delta)-DP')
parser.add_argument('--max_norm', type=float, default=0.1, help='clip per-sample gradients')
parser.add_argument('--sigma', type=float, default=1.0, help='noise multiplier')

parser.add_argument('--norm', type=str, default='group')
parser.add_argument('--num_groups', type=int, default=27)

args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print('Device: ', device, flush=True)

train_dataset, test_dataset = get_dataset(args)
train_labels = train_dataset.targets
test_labels = test_dataset.targets
if args.dataset == 'cifar':
    train_labels = torch.tensor(train_labels)
    test_labels = torch.tensor(test_labels)

train_features = torch.load(f'features/{args.dataset}_train.pt')
test_features = torch.load(f'features/{args.dataset}_test.pt')

if args.model == 'linear':
    train_features = train_features.view(train_features.size(0), -1)
    test_features = test_features.view(test_features.size(0), -1)
    model = ScatterLinear(train_features.shape[1], 10, args.num_groups).to(device)
elif args.model == 'cnn':
    model = ScatterCNN(args.dataset, train_features.shape[1], 10, args.num_groups).to(device)
else:
    raise ValueError('Unknown model: {}'.format(args.model))
model.to(device)

train_dataset = TensorDataset(train_features, train_labels)
test_dataset = TensorDataset(test_features, test_labels)
trainloader = DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum)
elif args.optimizer == 'adam':
    args.lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay)
criterion = torch.nn.NLLLoss().to(device)

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

# def test_inference(model, testloader, device):
#     model.eval()
#     loss, total, correct = 0.0, 0.0, 0.0

#     for batch_idx, (features, labels) in enumerate(testloader):
#         features, labels = features.to(device), labels.to(device)

#         outputs = model(features)
#         batch_loss = criterion(outputs, labels)
#         loss += batch_loss.item()

#         _, pred_labels = torch.max(outputs, 1)
#         pred_labels = pred_labels.view(-1)
#         correct += torch.sum(torch.eq(pred_labels, labels)).item()
#         total += len(labels)

#     accuracy = correct/total
#     return accuracy, loss

# for epoch in range(args.epochs):
#     model.train()
#     batch_loss = []
#     for batch_idx, (features, labels) in enumerate(trainloader):
#         features, labels = features.to(device), labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(features)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         if batch_idx % 50 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch+1, batch_idx * len(features), len(trainloader.dataset),
#                 100. * batch_idx / len(trainloader), loss.item()), flush=True)
#         batch_loss.append(loss.item())

#     loss_avg = sum(batch_loss)/len(batch_loss)
#     print('\nTrain loss:', loss_avg, flush=True)
#     epoch_loss.append(loss_avg)

#     test_acc, test_loss = test_inference(model, testloader, device)
#     print("Test Accuracy: {:.2f}%".format(100*test_acc), flush=True)
