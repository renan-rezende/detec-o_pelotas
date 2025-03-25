import cv2
import numpy as np
import os

# Variáveis globais para controle do desenho
drawing = False  # Indica se o mouse está sendo pressionado
mode = 'draw'  # 'draw' para adicionar bordas, 'erase' para apagar

# Função de callback do mouse
def refine_contours(event, x, y, flags, param):
    global drawing, mode, output, edges
    
    if event == cv2.EVENT_LBUTTONDOWN:  # Botão esquerdo pressionado
        drawing = True
        if mode == 'draw':
            cv2.circle(output, (x, y), 2, (255, 255, 255), -1)  # Desenha branco
            edges[y-2:y+3, x-2:x+3] = 255  # Atualiza a máscara
        elif mode == 'erase':
            cv2.circle(output, (x, y), 2, (0, 0, 0), -1)  # Apaga com preto
            edges[y-2:y+3, x-2:x+3] = 0  # Atualiza a máscara
    
    elif event == cv2.EVENT_MOUSEMOVE and drawing:  # Mouse movendo enquanto pressionado
        if mode == 'draw':
            cv2.circle(output, (x, y), 2, (255, 255, 255), -1)
            edges[y-2:y+3, x-2:x+3] = 255
        elif mode == 'erase':
            cv2.circle(output, (x, y), 2, (0, 0, 0), -1)
            edges[y-2:y+3, x-2:x+3] = 0
    
    elif event == cv2.EVENT_LBUTTONUP:  # Botão esquerdo solto
        drawing = False

# Define os diretórios
input_dir = "Original"
output_dir = "pelotas_borda"

# Cria o diretório de saída se não existir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Solicita o nome do arquivo ao usuário
print("Digite o nome do arquivo da imagem (ex.: frame_0071.jpg):")
image_file = input().strip()  # Remove espaços extras

# Monta o caminho completo da imagem de entrada
img_path = os.path.join(input_dir, image_file)

# Carrega a imagem original
img_original = cv2.imread(img_path)
if img_original is None:
    print(f"Erro ao carregar {img_path}. Verifique o nome ou o caminho e tente novamente.")
    exit()

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

# Abre a janela para refinamento manual
cv2.namedWindow("Refine Contours")
cv2.setMouseCallback("Refine Contours", refine_contours)

print("Instruções:")
print("- Clique e arraste com o botão esquerdo para desenhar bordas brancas.")
print("- Pressione 'e' e clique/arraste para apagar bordas (preto).")
print("- Pressione 's' para salvar e encerrar.")
print("- Pressione 'q' para sair sem salvar.")

while True:
    cv2.imshow("Refine Contours", output)
    key = cv2.waitKey(1) & 0xFF
    
    # Alterna entre modo de desenho e apagamento
    if key == ord('e'):
        mode = 'erase'
        print("Modo: Apagar")
    elif key == ord('d'):
        mode = 'draw'
        print("Modo: Desenhar")
    
    # Salva a imagem refinada
    elif key == ord('s'):
        output_path = os.path.join(output_dir, image_file)
        cv2.imwrite(output_path, output)
        print(f"Imagem salva em: {output_path}")
        break
    
    # Sai sem salvar
    elif key == ord('q'):
        print(f"Imagem {image_file} ignorada.")
        break

cv2.destroyAllWindows()

print("Processamento concluído!")