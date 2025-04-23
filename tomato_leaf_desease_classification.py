import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# define transformation 
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_dir = './PlantVillage'

# load data
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_transforms)

# filter classes for tomatoes
tomato_classes = [cls for cls in train_dataset.classes if cls.startswith('Tomato')]

# define class for filtering and renumeration of tomato images
class TomatoDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, tomato_classes):
        #create list of tomato classes indices from base dataset
        tomato_class_indices = [base_dataset.class_to_idx[cls] for cls in tomato_classes]
        self.samples = [s for s in base_dataset.samples if s[1] in tomato_class_indices]
        self.loader = base_dataset.loader
        self.transform = base_dataset.transform
        self.new_idx = {orig: i for i, orig in enumerate(tomato_class_indices)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, orig_label = self.samples[idx]
        img = self.loader(path)
        img = self.transform(img)
        new_label = self.new_idx[orig_label]
        return img, new_label

# selecting Tomato images
train_tomato_dataset = TomatoDataset(train_dataset, tomato_classes)
val_tomato_dataset = TomatoDataset(val_dataset, tomato_classes)

train_loader = DataLoader(train_tomato_dataset, batch_size=25, shuffle=True)
val_loader = DataLoader(val_tomato_dataset, batch_size=25, shuffle=False)

for images, labels in train_loader:
    print(f"Tomato Train Batch - size of images: {images.shape}, size of labels: {labels.shape}")
    break

for images, labels in val_loader:
    print(f"Tomato Val Batch - size of images: {images.shape}, size of labels: {labels.shape}")
    break


#VGG16
def create_vgg16_model(num_classes):
    model = models.vgg16(pretrained=True)
    # freezing convolutional layers
    for param in model.features.parameters():
        param.requires_grad = False
    # adaptation of the fully connected layer
    model.classifier[6] = nn.Linear(4096, num_classes)
    # dropout
    model.classifier = nn.Sequential(
        *model.classifier[:-1],
        nn.Dropout(0.25),
        model.classifier[-1]
    )
    return model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# model training
def train_model(model, train_loader, val_loader, num_epochs=25, lr=0.0001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_losses, train_accuracies, val_accuracies, val_losses = [], [], [], []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # Calculating training accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        model.eval()
        val_correct, val_total = 0, 0
        val_running_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_running_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    
    return train_accuracies, val_accuracies, train_losses, val_losses

# model evaluation
def evaluate_model(model, loader, dataset_name):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_labels, all_preds, all_probs = [], [], []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = nn.Softmax(dim=1)(outputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # metrics calculation
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    all_probs = np.array(all_probs)
    roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    mcc = matthews_corrcoef(all_labels, all_preds)
    
    print(f'\n{dataset_name} Evaluation Metrics:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')
    print(f'ROC-AUC: {roc_auc:.4f}')
    print(f'MCC: {mcc:.4f}')
    
    # confusion matrics
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# plot history visualisation
def plot_history(hist, title):
    tr_acc, vl_acc, tr_loss, vl_loss = hist
    epochs = range(1, len(tr_acc)+1)
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, tr_acc, 'b', label='Train Acc')
    plt.plot(epochs, vl_acc, 'r', label='Val Acc')
    plt.title(title+" Acc"); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(epochs, tr_loss, 'b', label='Train Loss')
    plt.plot(epochs, vl_loss, 'r', label='Val Loss')
    plt.title(title+" Loss"); plt.legend()
    plt.show()

# confusion matrix visualisation
def plot_confusion(model, loader, classes, title):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds=[]; labs=[]
    model.eval()
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            out = model(x)
            _,p = torch.max(out,1)
            preds += p.cpu().tolist()
            labs  += y.tolist()
    cm = confusion_matrix(labs, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(8,6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d', xticks_rotation='vertical')
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    num_classes = len(tomato_classes)
    print(f"Number of tomato classes: {num_classes}")
    
    model = create_vgg16_model(num_classes)
    
    print("\nTraining model for Tomatoes...")
    train_accuracies, val_accuracies, train_losses, val_losses = train_model(model, train_loader, val_loader)
    
    evaluate_model(model, val_loader, "Tomato Validation")
    
    plot_history([train_accuracies, val_accuracies, train_losses, val_losses], "Tomato Model")
    
    plot_confusion(model, val_loader, tomato_classes, "Tomato Confusion Matrix")