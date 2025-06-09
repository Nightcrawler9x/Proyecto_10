<div align="center">

# Deblurring de imagen utilizando Depth Maps
<br>

[Oscar Ortega](https://github.com/Nightcrawler9x) • [Jeferson Acevedo](https://github.com/Jeferson0809) • [Samuel Traslaviña](https://github.com/samtras)

---

</div>


El desenfoque por profundidad (*Depth of Field*, DoF) es un fenómeno común en fotografía y visión por computador, que afecta la nitidez de las imágenes y puede dificultar su análisis o procesamiento posterior.

Este proyecto se enfoca en la **recuperación de imágenes nítidas a partir de versiones borrosas** utilizando técnicas de deblurring asistidas por mapas de profundidad (*depth maps*). Estos mapas actúan como una guía estructural, permitiendo identificar qué zonas están más o menos desenfocadas según su distancia a la cámara.

> **Objetivo:** Desarrollar un modelo que combine imágenes RGB y mapas de profundidad para mejorar el proceso de restauración visual.

---
## Estructura del repositorio

- `data/` — Notebooks para la preparación y carga de datos.
- `images/` — Visualizaciones y resultados.
- `loss/` — Funciones de pérdida utilizadas en el entrenamiento.
- `models/` — Implementación de arquitecturas de modelos y archivos relacionados.
- `notebooks/` — Experimentos, pruebas en notebooks de Jupyter.
- `train/` — Scripts y lógica relacionados con el entrenamiento del modelo.
  
---

###  NYU Depth Dataset
🔗 [Enlace a Google Drive](https://drive.google.com/file/d/1gIjf8KDoUY-dObftp32AXrTgtaDEw4ek/view?usp=sharing)

Compuesto por 1.449 pares de imágenes RGB y de profundidad alineadas (resolución de 640 × 480) de 464 escenas interiores diversas, capturadas por el sensor Microsoft Kinect.

![image](https://github.com/user-attachments/assets/108e9709-4358-4e1d-8330-e1c7b13c2a0f)

---

### Slides  
https://www.canva.com/design/DAGpf_4pmaQ/4972YRulfo2jezUYWZmYqw/edit?utm_content=DAGpf_4pmaQ&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

https://drive.google.com/drive/folders/1AMCKyZD803UZ2YjOGmrm1tgSKhaU3YzF?usp=sharing

---




