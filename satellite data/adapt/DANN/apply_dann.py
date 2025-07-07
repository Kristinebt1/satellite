import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class DANN(nn.Module):
    def __init__(self, num_classes):
        super(DANN, self).__init__()
        # Feature Extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.label_predictor = nn.Sequential(
            nn.Linear(50 * 4 * 4, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes)
        )
        
        self.domain_classifier = nn.Sequential(
            nn.Linear(50 * 4 * 4, 100),
            nn.ReLU(),
            nn.Linear(100, 2)  
        )
        self.grl = GradientReversalLayer()

    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        class_output = self.label_predictor(features)
        domain_output = self.domain_classifier(self.grl(features))
        return class_output, domain_output


def train_dann(model, source_loader, target_loader, optimizer, criterion, domain_criterion, device, num_epochs=10):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for (source_data, source_labels), (target_data, _) in zip(source_loader, target_loader):
            source_data, source_labels = source_data.to(device), source_labels.to(device)
            target_data = target_data.to(device)

           
            data = torch.cat((source_data, target_data), 0)
            domain_labels = torch.cat((
                torch.zeros(source_data.size(0)).long(),  
                torch.ones(target_data.size(0)).long()   
            )).to(device)

           
            class_output, domain_output = model(data)

            
            classification_loss = criterion(class_output[:source_data.size(0)], source_labels)
            domain_loss = domain_criterion(domain_output, domain_labels)
            loss = classification_loss + domain_loss

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")


if __name__ == "__main__":
    
    batch_size = 32
    num_classes = 10
    num_epochs = 10
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    source_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    target_dataset = datasets.SVHN(root="./data", split="train", transform=transform, download=True)
    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True)

    
    model = DANN(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    
    train_dann(model, source_loader, target_loader, optimizer, criterion, domain_criterion, device, num_epochs)
