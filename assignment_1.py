# === Step 1: Setup & Installation ===
!pip install wandb torch torchvision --quiet

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time, numpy as np, glob, zipfile

wandb.login()

# === Step 2: Dataset Preparation ===

# CIFAR-10 normalization and augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)

# === Step 3: Model Definition (VGG6) ===
cfg_vgg6 = [64, 'M', 128, 'M', 256, 'M']

def make_layers(cfg, batch_norm=False, activation_fn=nn.ReLU):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), activation_fn()]
            else:
                layers += [conv2d, activation_fn()]
            in_channels = v
    return nn.Sequential(*layers)

class VGG(nn.Module):
    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def vgg(cfg, num_classes=10, batch_norm=True, activation_fn=nn.ReLU):
    return VGG(make_layers(cfg, batch_norm=batch_norm, activation_fn=activation_fn),
               num_classes=num_classes)

# === Step 4: Training & Evaluation ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_one_config(config=None):
    with wandb.init(config=config) as run:
        config = wandb.config
        activation_dict = {"ReLU": nn.ReLU, "SiLU": nn.SiLU, "GELU": nn.GELU}
        activation_fn = activation_dict[config.activation]

        trainloader = DataLoader(trainset, batch_size=config.batch_size,
                                 shuffle=True, num_workers=2)
        testloader = DataLoader(testset, batch_size=100,
                                shuffle=False, num_workers=2)

        model = vgg(cfg_vgg6, batch_norm=True, activation_fn=activation_fn).to(device)
        criterion = nn.CrossEntropyLoss()

        if config.optimizer == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        elif config.optimizer == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=config.learning_rate,
                                  momentum=0.9, nesterov=True)
        else:
            optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)

        best_val_acc, best_model_path = 0.0, f"best_model_{run.id}.pth"

        for epoch in range(config.epochs):
            model.train()
            total, correct, running_loss = 0, 0, 0.0
            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            train_acc = 100. * correct / total
            train_loss = running_loss / total

            # Validation
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, targets in testloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()

            val_loss /= val_total
            val_acc = 100. * val_correct / val_total

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            })

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
                wandb.save(best_model_path)
                artifact = wandb.Artifact(name=f"vgg6-best-{run.id}", type="model")
                artifact.add_file(best_model_path)
                run.log_artifact(artifact)

        print(f"Run {run.id} finished. Best val_acc = {best_val_acc:.2f}. Saved: {best_model_path}")

# === Step 5: Sweep Configuration & Execution ===
sweep_config = {
    'method': 'random',
    'metric': {'name': 'val_acc', 'goal': 'maximize'},
    'parameters': {
        'activation': {'values': ['ReLU', 'SiLU', 'GELU']},
        'optimizer': {'values': ['Adam', 'SGD', 'RMSprop']},
        'learning_rate': {'values': [0.001, 0.01]},
        'batch_size': {'values': [64, 128]},
        'epochs': {'values': [10, 20]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="VGG6_CIFAR10_Assignment_Final")
wandb.agent(sweep_id, function=train_one_config, count=15)

# === Step 6: Collect & Download Best Models ===
saved_files = glob.glob("best_model_*.pth")
if saved_files:
    zip_name = "best_models_collected.zip"
    with zipfile.ZipFile(zip_name, 'w') as zf:
        for f in saved_files:
            zf.write(f)
    print(f"Created archive: {zip_name} containing {len(saved_files)} model(s).")

    from google.colab import files as colab_files
    colab_files.download(zip_name)
else:
    print("No best_model_*.pth files found.")