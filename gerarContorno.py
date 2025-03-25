import cv2
import numpy as np

# Carrega a imagem original
img_original = cv2.imread("Original/frame_0071.jpg")  # Carrega em BGR (colorida)
img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)  # Converte para escala de cinza

# Aplica desfoque para reduzir ruído
img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

# Detecta bordas com Canny
edges = cv2.Canny(img_blur, 80, 160)

# Filtra bordas pequenas
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros_like(edges)  # Máscara para as bordas filtradas
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 20:  # Ajuste este valor para o tamanho mínimo desejado
        cv2.drawContours(mask, [contour], -1, 255, thickness=1)  # Espessura da borda = 1 pixel
edges = mask

# Converte as bordas para uma imagem com 3 canais (para combinar com a original)
edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

# Sobrepoe as bordas brancas na imagem original
output = img_original.copy()  # Cria uma cópia da imagem original
output[edges != 0] = [255, 255, 255]  # Define as bordas como branco na imagem original

# Salva a imagem resultante
cv2.imwrite("pelota1_edges_refined.png", output)

