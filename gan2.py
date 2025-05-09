import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# === Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform),
    batch_size=64, shuffle=True
)

# === Generator Model ===
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Tanh()  # Output range [-1, 1]
        )

    def forward(self, x):
        x = self.model(x)
        return x.view(-1, 1, 28, 28)

# === Discriminator Model ===
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten image
        return self.model(x)

# === Initialize Models ===
G = Generator().to(device)
D = Discriminator().to(device)

# === Loss and Optimizers ===
loss_fn = nn.BCELoss()
opt_G = torch.optim.Adam(G.parameters(), lr=0.0002)
opt_D = torch.optim.Adam(D.parameters(), lr=0.0002)

# === Training Loop ===
epochs = 5

for epoch in range(epochs):
    for real_imgs, _ in train_loader:
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)

        # === Train Discriminator ===
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        outputs = D(real_imgs)
        d_loss_real = loss_fn(outputs, real_labels)

        noise = torch.randn(batch_size, 100).to(device)
        fake_imgs = G(noise)
        outputs = D(fake_imgs.detach())
        d_loss_fake = loss_fn(outputs, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        # === Train Generator ===
        noise = torch.randn(batch_size, 100).to(device)
        fake_imgs = G(noise)
        outputs = D(fake_imgs)
        g_loss = loss_fn(outputs, real_labels)  # Trick D into thinking fakes are real

        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

    print(f"Epoch {epoch+1}/{epochs} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

# === Generate Sample Images ===
G.eval()
sample_noise = torch.randn(16, 100).to(device)
fake_images = G(sample_noise).detach().cpu()

# Show samples
grid = torchvision.utils.make_grid(fake_images, nrow=4, normalize=True)
plt.imshow(grid.permute(1, 2, 0))
plt.title("Generated Digits")
plt.axis("off")
plt.show()
