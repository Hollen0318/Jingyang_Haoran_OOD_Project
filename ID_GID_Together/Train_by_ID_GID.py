# %% [markdown]
# # Libraries

# %%
import torch
from torch.utils.data import Subset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from tqdm import tqdm

# %% [markdown]
# # Functions

# %%
def get_subset_indices(dataset, percentage, seed=0):
  rng = np.random.RandomState(seed)
  # data = dataset.data
  targets = np.array(dataset.targets)
  num_classes = len(np.unique(targets))
  num_samples_per_class = int(percentage*len(dataset)/num_classes)

  indices = []

  for c in range(num_classes):
    class_indices = (targets == c).nonzero()[0]
    indices.extend(
        list(rng.choice(class_indices, size=num_samples_per_class, replace=False))
    )
  return indices

# %%
def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * \
                (1 + np.cos(step / total_steps * np.pi))

# %%
def dataloader_with_seed_perc(trainset, seed, perc):
  train_subset = Subset(
      trainset, 
      get_subset_indices(trainset, perc, int(seed))
  )
  
  trainloader = torch.utils.data.DataLoader(
    train_subset, batch_size=128, shuffle=True, num_workers=2)

  return trainloader

# %%
def optimizer_with_lr(net, lr):
  optimizer = torch.optim.SGD(
    net.parameters(),
    lr,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
)
  return optimizer

# %%
def schedular_with_lr(EPOCHS, opt, lr, trainloader):
  scheduler = torch.optim.lr_scheduler.LambdaLR(
    opt,
    lr_lambda=lambda step: cosine_annealing(
        step,
        EPOCHS * len(trainloader),
        1,
        1e-6 / lr,
    ),
)
  return scheduler

# %% [markdown]
# # Neural Network

# %%
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,
                               self.expansion * planes,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18_32x32(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=None, num_classes=10):
        super(ResNet18_32x32, self).__init__()
        if num_blocks is None:
            num_blocks = [2, 2, 2, 2]
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3,
                               64,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.avgpool = nn.AvgPool2d(4)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.feature_size = 512 * block.expansion

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_feature=False, return_feature_list=False):
        feature1 = F.relu(self.bn1(self.conv1(x)))
        feature2 = self.layer1(feature1)
        feature3 = self.layer2(feature2)
        feature4 = self.layer3(feature3)
        feature5 = self.layer4(feature4)
        feature5 = self.avgpool(feature5)
        feature = feature5.view(feature5.size(0), -1)
        logits_cls = self.fc(feature)
        feature_list = [feature1, feature2, feature3, feature4, feature5]
        if return_feature:
            return logits_cls, feature
        elif return_feature_list:
            return logits_cls, feature_list
        else:
            return logits_cls

    def forward_threshold(self, x, threshold):
        feature1 = F.relu(self.bn1(self.conv1(x)))
        feature2 = self.layer1(feature1)
        feature3 = self.layer2(feature2)
        feature4 = self.layer3(feature3)
        feature5 = self.layer4(feature4)
        feature5 = self.avgpool(feature5)
        feature = feature5.clip(max=threshold)
        feature = feature.view(feature.size(0), -1)
        logits_cls = self.fc(feature)

        return logits_cls

    def get_fc(self):
        fc = self.fc
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()


# %% [markdown]
# # Datasets

# %% [markdown]
# ## Concatenate

# %%
class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

# %% [markdown]
# ## Transform

# %%
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# %% [markdown]
# ## Read Images

# %%
GID_data_path = "/home/hz271/PyTorch-StudioGAN/biggan_images/samples/CIFAR10-BigGAN-DiffAug-train-2022_02_11_07_23_15/fake"

# Datasets
generated_train_dataset = torchvision.datasets.ImageFolder(
    root=GID_data_path,
    transform=transform_train
)

original_train_dataset = torchvision.datasets.CIFAR10(
    root='/home/hz271/OOD/Saved/data/', train=True, download=True, transform=transform_train)

# Concatenate
train_set = torch.utils.data.ConcatDataset(
    [generated_train_dataset, original_train_dataset]
)

test_set = torchvision.datasets.CIFAR10(
    root='/home/hz271/OOD/Saved/data/', train=False, download=True, transform=transform_test)

# %% [markdown]
# ## Dataloder

# %%
trainloader = torch.utils.data.DataLoader(train_set, batch_size = 256, shuffle = True, num_workers = 2)

test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=100, shuffle=False, num_workers=2)

# %% [markdown]
# # Checkpoint

# %%
import os
CHECKPOINT_FOLDER = "/home/hz271/OOD/Saved/id_gid_saved_model"

# %% [markdown]
# # Train

# %%
# trainloader = dataloader_with_seed_perc(train_set, 0, 0.1)
for batch_idx, (inputs, targets) in enumerate(trainloader):
    print("batch_idx = ", batch_idx, " inputs = ", inputs)
    break;

# %%
criterion = torch.nn.CrossEntropyLoss().to(device)
EPOCHS = 200
best_acc_list = []
for SEED_index in range(3):
  for perc_index in range(10,0,-1):
    perc = perc_index/10.0
    lr = 0.1/perc
    net = ResNet18_32x32().to(device)
    # trainloader = dataloader_with_seed_perc(train_set, SEED_index, perc)
    optimizer = optimizer_with_lr(net, lr)
    schedular = schedular_with_lr(EPOCHS, optimizer, lr, trainloader)
    # print(optimizer.param_groups)
    current_learning_rate = lr
    best_acc = 0
    for i in range(EPOCHS):
        # if i % 3 == 0 and i != 0:
        #     current_learning_rate = current_learning_rate * 0.95
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = current_learning_rate
        #     print("Current learning rate has decayed to %f" %current_learning_rate)
        net.train()

        print("Epoch %d:" %i)
        # this help you compute the training accuracy
        total_examples = 0
        correct_examples = 0

        train_loss = 0 # track training loss if you want
        
        # Train the model for 1 epoch.
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            ####################################
            # your code here
            # copy inputs to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            # zero the gradient
            optimizer.zero_grad()
            # compute the output and loss
            outputs = net(inputs)
            loss = criterion(outputs,targets)
            # l1_lambda = 0.001
            # l1_norm = sum(p.abs().sum()
            #   for p in model.parameters())
            # loss = loss + l1_lambda * l1_norm
            # backpropagation
            loss.backward()
            # apply gradient and update the weights
            optimizer.step()
            schedular.step()
            # count the number of correctly predicted samples in the current batch
            _, predicted = torch.max(outputs, 1)
            # print("targets:", targets, targets.shape)
            # print("predicted:", predicted, predicted.shape)
            correct = predicted.eq(targets).sum()
            correct_examples += correct
            total_examples += targets.shape[0]
            train_loss += loss
            ####################################
                    
        avg_loss = train_loss / len(trainloader)
        avg_acc = correct_examples / total_examples
        print("Training loss: %.4f, Training accuracy: %.4f" %(avg_loss, avg_acc))
        if avg_acc > best_acc:
            best_acc = avg_acc
            if not os.path.exists(CHECKPOINT_FOLDER):
                os.makedirs(CHECKPOINT_FOLDER)
            torch.save(net.state_dict(), os.path.join(CHECKPOINT_FOLDER,"seed_"+str(SEED_index)+"_perc_"+str(perc)+".pth"))
    print("="*50)
    print(f"==> Optimization finished! Best validation accuracy: {best_acc:.4f}")
    best_acc_list.append(best_acc)