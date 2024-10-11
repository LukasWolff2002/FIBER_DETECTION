import torch
import torch.nn as nn  # Esta línea es la clave para evitar el error
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

# Definir el mismo modelo (U-Net o el que hayas usado en el entrenamiento)
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

# Función para seleccionar y procesar una imagen
def seleccionar_imagen(directorio_imagenes):
    imagenes_disponibles = [f for f in os.listdir(directorio_imagenes) if f.endswith(('jpg', 'jpeg', 'png'))]

    if len(imagenes_disponibles) == 0:
        raise Exception("No se encontraron imágenes en el directorio.")

    print("Imágenes disponibles:")
    for i, nombre_imagen in enumerate(imagenes_disponibles):
        print(f"{i}: {nombre_imagen}")

    indice = int(input("Selecciona el índice de la imagen que deseas usar: "))
    imagen_seleccionada = imagenes_disponibles[indice]

    ruta_imagen = os.path.join(directorio_imagenes, imagen_seleccionada)
    imagen = Image.open(ruta_imagen).convert('RGB')

    return imagen

# Procesar la imagen seleccionada
def procesar_imagen(imagen):
    transformacion = transforms.Compose([
        transforms.Resize((256, 256)),  # Cambiar el tamaño a 256x256
        transforms.ToTensor()  # Convertir a tensor
    ])
    
    return transformacion(imagen)

# Visualizar las predicciones
def visualizar_predicciones(imagen, mascara_predicha):
    imagen = imagen.cpu().numpy()
    mascara_predicha = mascara_predicha.squeeze().cpu().numpy()

    plt.subplot(1, 2, 1)
    plt.title('Imagen')
    plt.imshow(np.transpose(imagen, (1, 2, 0)))
    
    plt.subplot(1, 2, 2)
    plt.title('Máscara predicha')
    plt.imshow(mascara_predicha, cmap='gray')
    
    plt.show()

# Cargar el modelo y las imágenes
if __name__ == "__main__":
    # Inicializar el modelo
    model = UNet().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Cargar los pesos guardados
    model.load_state_dict(torch.load("modelo_sam_entrenado.pth"))
    model.eval()

    # Directorio de las imágenes de prueba
    directorio_imagenes = "TEST"

    # Seleccionar una imagen del directorio
    imagen_seleccionada = seleccionar_imagen(directorio_imagenes)

    # Procesar la imagen
    imagen_procesada = procesar_imagen(imagen_seleccionada)

    # Mover la imagen a la GPU si está disponible
    imagen_procesada = imagen_procesada.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Pasar la imagen al modelo y obtener las predicciones
    with torch.no_grad():
        predicciones = model(imagen_procesada.unsqueeze(0))

    # Visualizar la imagen y la máscara predicha
    visualizar_predicciones(imagen_procesada, predicciones)
