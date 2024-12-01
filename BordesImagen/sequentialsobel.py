import cv2
import numpy as np
import time

# Configuración para deshabilitar procesamiento paralelo
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)

# Filtros Sobel
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

def sobel_sequential(image):
    # Aplicar convoluciones Sobel
    gx = cv2.filter2D(image, -1, sobel_x)  # Gradiente horizontal
    gy = cv2.filter2D(image, -1, sobel_y)  # Gradiente vertical
    # Calcular magnitud del gradiente
    gradient = np.sqrt(gx**2 + gy**2)
    gradient = (gradient / gradient.max()) * 255  # Normalizar a [0, 255]
    return gradient.astype(np.uint8)

if __name__ == "__main__":
    # Leer imagen en escala de grises
    image = cv2.imread('BordesImagen/image.jpg', cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: No se pudo cargar la imagen.")
        exit(1)

    # Medir tiempo de procesamiento
    start_time = time.time()
    result = sobel_sequential(image)
    elapsed_time = time.time() - start_time

    # Guardar y mostrar resultados
    cv2.imwrite('resultados/sequential_filtered.jpg', result)
    print(f"Tiempo Secuencial (sin paralelismo): {elapsed_time:.4f} segundos")
