import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super(Attention, self).__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, dropout=0.1):
        super(ViTBlock, self).__init__()
        self.attention = Attention(dim, heads=num_heads, dropout=dropout)
        self.mlp = MLP(dim, mlp_dim, dropout=dropout)

    def forward(self, x):
        x = x + self.attention(x)
        x = x + self.mlp(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3, dropout=0.1):
        super(VisionTransformer, self).__init__()
        assert image_size % patch_size == 0, "image dimensions must be divisible by the patch size"
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        self.patch_embedding = nn.Conv2d(in_channels=channels, out_channels=dim, kernel_size=patch_size, stride=patch_size)
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)
        self.transformer = nn.Sequential(*[ViTBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)])
        self.to_cls_token = nn.Identity()
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        p = self.patch_embedding(x)
        b, _, _, _ = p.shape
        p = rearrange(p, 'b d h w -> b (h w) d')
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, p), dim=1)
        x += self.positional_embedding
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.to_cls_token(x[:, 0])
        return self.fc(x)

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Define transformations for preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),           # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet mean and std
])

# Load the dataset
dataset = ImageFolder(root='./content', transform=transform)

# Define data loader
batch_size = 32
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Instantiate the VisionTransformer model
image_size = 224  # Assuming resized images are 224x224
patch_size = 16   # Patch size used in the ViT model
num_classes = len(dataset.classes)  # Number of classes in the dataset
dim = 512         # Dimension of the embeddings
depth = 6         # Number of transformer blocks
heads = 8         # Number of attention heads
mlp_dim = 1024    # Dimension of the feedforward layers
vit_model = VisionTransformer(image_size, patch_size, num_classes, dim, depth, heads, mlp_dim)

# Define optimizer and loss function
optimizer = torch.optim.Adam(vit_model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    vit_model.train()
    for images, labels in data_loader:
        optimizer.zero_grad()
        outputs = vit_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    break

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Assuming you have already loaded your model and dataset
# Also assuming you have defined your criterion (loss function)

# Set model to evaluation mode
vit_model.eval()

# Initialize lists to store predictions and labels
all_predictions = []
all_labels = []

# Iterate over the dataset
with torch.no_grad():
    for images, labels in data_loader:
        # Forward pass
        outputs = vit_model(images)
        _, predicted = torch.max(outputs, 1)

        # Store predictions and labels
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Compute metrics
accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, average='macro')
recall = recall_score(all_labels, all_predictions, average='macro')
f1 = f1_score(all_labels, all_predictions, average='macro')

# Print metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
