#git clone https://github.com/ultralytics/yolov5
#cd yolov5
#pip install -r requirements.txt

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import cv2
import numpy as np
import torchvision.transforms as transforms

# Cargar el modelo YOLOv5 preentrenado
def cargar_modelo_yolo():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # Modelo YOLOv5 pequeño preentrenado
    # Congelar todas las capas excepto las últimas
    for param in model.parameters():
        param.requires_grad = False
    
    # Descongelar solo la última capa
    model.model[-1].requires_grad = True
    
    return model

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

# Función de entrenamiento con fine-tuning
def entrenar_modelo(modelo, dataloader, epochs=10, lr=1e-4):
    modelo.train()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, modelo.parameters()), lr=lr)  # Solo ajustar las capas necesarias
    criterio = torch.nn.BCELoss()

    for epoch in range(epochs):
        for batch_idx, (imagenes, mascaras) in enumerate(dataloader):
            imagenes = imagenes.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            mascaras = mascaras.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            # YOLOv5 espera imágenes sin máscaras, ajusta este código según tu necesidad
            predicciones = modelo(imagenes)  # Para YOLO, debes ajustar la salida

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
        transforms.Resize((640, 640))  # Tamaño recomendado para YOLOv5
    ])

    dataset = ImagenesConMascaras(image_dir, mask_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Cargar el modelo YOLOv5 con fine-tuning
    model = cargar_modelo_yolo().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Entrenar el modelo
    entrenar_modelo(model, dataloader, epochs=10)

    # Guardar los pesos del modelo
    torch.save(model.state_dict(), "modelo_yolo_finetuned.pth")
    print("Modelo guardado exitosamente en 'modelo_yolo_finetuned.pth'.")
