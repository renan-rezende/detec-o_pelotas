import cv2
import numpy as np
import os

# Define os diretórios
input_dir = "Original"
output_dir = "pelotas_borda2"

# Cria o diretório de saída se não existir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Lista todas as imagens na pasta "Original"
image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Processa cada imagem
for image_file in image_files:
    # Monta o caminho completo da imagem de entrada
    img_path = os.path.join(input_dir, image_file)
    
    # Carrega a imagem original
    img_original = cv2.imread(img_path)  # Carrega em BGR (colorida)
    if img_original is None:
        print(f"Erro ao carregar {img_path}. Pulando...")
        continue
    
    img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

    # Aplica desfoque para reduzir ruído
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # Detecta bordas com Canny
    edges = cv2.Canny(img_blur, 80, 160)

    # Filtra bordas pequenas
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(edges)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 20:
            cv2.drawContours(mask, [contour], -1, 255, thickness=1)
    edges = mask

    # Converte as bordas para uma imagem com 3 canais
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Sobrepoe as bordas brancas na imagem original
    output = img_original.copy()
    output[edges != 0] = [255, 255, 255]

    # Define as coordenadas do recorte retangular (exemplo: canto superior esquerdo e inferior direito)
    y1, y2 = 0, 992  # Altura (linhas)
    x1, x2 = 257, 1448  # Largura (colunas)
    output_cropped = output[y1:y2, x1:x2]  # Recorta a região especificada
    edges_cropped = edges[y1:y2, x1:x2]    # Recorta a máscara também (opcional)

    # Monta o caminho de saída com o mesmo nome do arquivo original
    output_path = os.path.join(output_dir, image_file)

    # Salva a imagem recortada
    cv2.imwrite(output_path, output_cropped)
    print(f"Imagem recortada e salva em: {output_path}")

