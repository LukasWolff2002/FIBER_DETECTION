import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import cv2
import numpy as np
import torchvision.transforms as transforms

# Definir el modelo U-Net o cualquier otro modelo que estés usando
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=1)  # Salida binaria (1 canal)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = torch.relu(x1)
        x1 = self.pool(x1)
        x2 = self.conv2(x1)
        x2 = torch.relu(x2)
        x3 = self.upconv1(x2)
        x3 = torch.relu(x3)
        output = self.conv3(x3)
        return torch.sigmoid(output)

# Definir el dataset (puedes ajustarlo según tus datos)
class ImagenesConMascaras(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = os.listdir(image_dir)
        self.mask_files = os.listdir(mask_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        mask = (mask > 128).astype(np.float32)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# Función de entrenamiento
def entrenar_modelo(modelo, dataloader, epochs=10, lr=1e-4):
    modelo.train()
    optimizer = optim.Adam(modelo.parameters(), lr=lr)
    criterio = nn.BCELoss()

    for epoch in range(epochs):
        for batch_idx, (imagenes, mascaras) in enumerate(dataloader):
            imagenes = imagenes.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            mascaras = mascaras.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            predicciones = modelo(imagenes)
            loss = criterio(predicciones, mascaras)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item()}')

# Definir los parámetros y entrenar
if __name__ == "__main__":
    image_dir = "DATA/images"
    mask_dir = "DATA/masks"
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256))
    ])

    dataset = ImagenesConMascaras(image_dir, mask_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Inicializar el modelo
    model = UNet().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Entrenar el modelo
    entrenar_modelo(model, dataloader, epochs=10)

    # Guardar los pesos del modelo
    torch.save(model.state_dict(), "modelo_sam_entrenado.pth")
    print("Modelo guardado exitosamente en 'modelo_sam_entrenado.pth'.")
