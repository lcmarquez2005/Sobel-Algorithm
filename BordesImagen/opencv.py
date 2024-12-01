import numpy as np
import cv2
import time

# Filtros Sobel
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

def sobel_sequential(image):
    # Convertir a escala de grises si la imagen es en color
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplicar filtro Gaussiano para suavizar la imagen y reducir el ruido
    image = cv2.GaussianBlur(image, (5, 5), 0)  # Tamaño de kernel de 5x5
    
    # Aplicar convoluciones Sobel
    gx = cv2.filter2D(image, -1, sobel_x)
    gy = cv2.filter2D(image, -1, sobel_y)
    
    # Calcular magnitud del gradiente
    gradient = np.sqrt(gx**2 + gy**2)
    # Aumentar el brillo multiplicando por un factor
    gradient = gradient * 5  # Factor de aumento de brillo
    # Recortar para que los valores no excedan 255
    gradient = np.clip(gradient, 0, 255)
    
    return gradient.astype(np.uint8)





# Leer imagen
image = cv2.imread('BordesImagen/image.jpg', cv2.IMREAD_COLOR)
if image is None:
    print("Error: No se pudo cargar la imagen.")
    exit(1)
else:
    print("Imagen cargada correctamente.")




# Medir tiempo
start_time = time.time()
result = sobel_sequential(image)
elapsed_time = time.time() - start_time




# Guardar y mostrar resultados
cv2.imwrite('resultados/result_sequential_brightness_smoothed.jpg', result)
print(f"Tiempo Secuencial: {elapsed_time:.4f} segundos")



# Mostrar la imagen resultante
cv2.imshow('Bordes con Sobel, Brillo Aumentado y Suavizado', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
