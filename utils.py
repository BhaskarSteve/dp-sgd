import numpy as np
from torchvision import datasets, transforms

def compute_scattering_features(images, scattering, batch_size, device):
    scattering_features = []
    num_samples = images.size(0)
    for i in range(0, num_samples, batch_size):
        if i % (batch_size*100) == 0:
            print('Processing samples [{}/{}]'.format(i, num_samples))
        batch_images = images[i:i+batch_size].to(device)
        if images.size(2) == 28: # MNIST
            Sx = scattering(batch_images)
            Sx = Sx.cpu().numpy()
            # Flatten the features (for linear models)
            # Sx = Sx.reshape(Sx.shape[0], -1)
        
            # Squeeze the batch dimension (for CNN models)
            Sx = Sx.squeeze(1)
        elif images.size(2) == 32: # CIFAR10
            Sx = scattering(batch_images).view(batch_images.size(0), 243, 8, 8)
            Sx = Sx.cpu().numpy()
        scattering_features.append(Sx)
    scattering_features = np.concatenate(scattering_features, axis=0)
    return scattering_features

def get_dataset(args):
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
    return train_dataset, test_dataset