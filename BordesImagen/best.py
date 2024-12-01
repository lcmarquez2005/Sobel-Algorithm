import numpy as np
import cv2
import time

# Definir máscaras Sobel
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])



# Función para realizar la convolución Sobel manualmente
def sobel_convolution(image):
    """Aplica la convolución Sobel para detectar bordes."""
    height, width = image.shape
    gx = np.zeros_like(image, dtype=np.float32)
    gy = np.zeros_like(image, dtype=np.float32)

    # Realizar la convolución Sobel
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            region = image[i - 1:i + 2, j - 1:j + 2]
            gx[i, j] = np.sum(region * sobel_x)
            gy[i, j] = np.sum(region * sobel_y)

    # Calcular la magnitud del gradiente
    gradient = np.sqrt(gx**2 + gy**2)
    gradient = np.clip(gradient, 0, 255)  # Asegurarse de que los valores estén en el rango [0, 255]
    return gradient.astype(np.uint8)



# Leer imagen en escala de grises
image = cv2.imread('BordesImagen/image.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error: No se pudo cargar la imagen.")
    exit(1)

# Paso 1: Aplicar filtro de desenfoque (blur) con kernel 5x5
image_blurred = cv2.blur(image, (3, 3))

# Paso 2: Aplicar la convolución Sobel para detectar bordes
start_time = time.time()
result = sobel_convolution(image_blurred)
elapsed_time = time.time() - start_time

# Guardar y mostrar resultados
cv2.imwrite('resultadosSequential/sequential_blurred_sobel.jpg', result)
print(f"Tiempo de procesamiento: {elapsed_time:.4f} segundos")
