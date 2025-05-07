# Detección y Explicación de Anomalías en Imágenes de Fondo de Ojo con Modelos de Lenguaje Visual (VLM)

Este proyecto tiene como objetivo desarrollar un sistema automatizado para la detección de patologías en imágenes de fondo de ojo (retinografías) utilizando modelos de lenguaje visual (VLM). Las patologías a detectar incluyen **retinopatía diabética**, **glaucoma** y otras anomalías relacionadas.

## Descripción

El proyecto usa un conjunto de datos de imágenes retinianas que fueron etiquetadas por un clínico en una escala del 0 al 4, donde:

- **0**: No DR (No Retinopatía Diabética)
- **1**: Mild (Leve)
- **2**: Moderate (Moderada)
- **3**: Severe (Severa)
- **4**: Proliferative DR (Retinopatía Proliferativa)

El objetivo es entrenar un modelo de **Red Neuronal Convolucional (CNN)** para clasificar las imágenes en las diferentes categorías mencionadas anteriormente.

## Requisitos

Para ejecutar este proyecto en tu máquina local, asegúrate de tener instalados los siguientes paquetes:

- Python 3.x
- Pytorch
- torchvision
- pandas
- Pillow
- tqdm

Puedes instalar las dependencias necesarias utilizando `pip`:

```bash
pip install torch torchvision pandas pillow tqdm
