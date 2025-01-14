#YOLO Object Detection com Transfer Learning
#Configuração do Ambiente

!pip install opencv-python
!pip install numpy
!pip install torch torchvision
!pip install labelme
  
#Preparação dos Dados
import os
import cv2
import numpy as np
from PIL import Image

def prepare_dataset(data_dir):
    # Criação das pastas necessárias
    if not os.path.exists('dataset'):
        os.makedirs('dataset/images')
        os.makedirs('dataset/labels')
    
    # Processamento das imagens e anotações
    for img_file in os.listdir(data_dir):
        if img_file.endswith('.jpg'):
            img_path = os.path.join(data_dir, img_file)
            # Copiar imagem para pasta de dataset
            img = Image.open(img_path)
            img.save(f'dataset/images/{img_file}')
  
#Transfer Learning com YOLO
import torch
from models import *
from utils.datasets import *
from utils.utils import *

def train_yolo():
    # Configurações do modelo
    cfg = 'cfg/yolov3.cfg'
    weights = 'weights/yolov3.weights'
    
    # Carrega modelo pré-treinado
    model = Darknet(cfg)
    model.load_weights(weights)
    
    # Modifica última camada para novas classes
    num_classes = 2  # Número de novas classes
    model.hyperparams['classes'] = num_classes
    
    # Configura otimizador
    optimizer = torch.optim.Adam(model.parameters())
    
    # Loop de treinamento
    for epoch in range(100):
        model.train()
        for batch_i, (imgs, targets) in enumerate(dataloader):
            loss = model(imgs)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
  
#Inferência e Detecção
def detect_objects(image_path, model):
    # Carrega e pré-processa a imagem
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Realiza a detecção
    results = model(img)
    
    # Desenha as bounding boxes
    for det in results.pred[0]:
        if det is not None and len(det):
            for *xyxy, conf, cls in det:
                label = f'{model.names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img, label=label)
    
    return img

# Exemplo de uso
image_path = 'test_image.jpg'
result_img = detect_objects(image_path, model)
cv2.imshow('Detection Result', result_img)
cv2.waitKey(0)
