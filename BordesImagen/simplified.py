import numpy as np
import cv2
import time
import multiprocessing as mp

# Definir máscaras Sobel
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

def sobel_convolution_segment(args):
    """
    Aplica la convolución Sobel en un segmento de la imagen.
    
    Args:
    - args: tupla (segment, start_row, end_row)
    
    Returns:
    - gx: gradiente en dirección x del segmento.
    - gy: gradiente en dirección y del segmento.
    - start_row: fila inicial del segmento.
    """
    segment, start_row, end_row = args
    height, width = segment.shape
    
    gx = np.zeros_like(segment, dtype=np.float32)
    gy = np.zeros_like(segment, dtype=np.float32)

    # Aplicar Sobel en cada píxel del segmento
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            region = segment[i - 1:i + 2, j - 1:j + 2]
            gx[i, j] = np.sum(region * sobel_x)
            gy[i, j] = np.sum(region * sobel_y)

    return gx, gy, start_row





def split_image(image, num_segments):
    """
    Divide la imagen en segmentos con superposición.

    Args:
    - image: imagen de entrada.
    - num_segments: número de segmentos.

    Returns:
    - lista de tuplas (segmento, start_row, end_row).
    """
    height, width = image.shape
    segment_height = height // num_segments
    segments = []

    for i in range(num_segments):
        start_row = max(0, i * segment_height - 1)  # Superposición superior
        end_row = min(height, (i + 1) * segment_height + 1)  # Superposición inferior
        segment = image[start_row:end_row, :]
        segments.append((segment, start_row, end_row))
    
    return segments






def reconstruir_imagen(resultados, gx_completo, gy_completo):
    """
    Reconstruye la imagen completa a partir de los gradientes calculados en cada segmento.

    Args:
    - resultados: lista de tuplas (gx, gy, start_row).
    - gx_completo: matriz inicializada para almacenar el gradiente completo en dirección x.
    - gy_completo: matriz inicializada para almacenar el gradiente completo en dirección y.
    """
    for index, (gx, gy, start_row) in enumerate(resultados):
        segmento_altura = gx.shape[0]
        
        if index == 0:  # Primer segmento
            gx_completo[start_row:start_row + segmento_altura - 1] = gx[:-1]
            gy_completo[start_row:start_row + segmento_altura - 1] = gy[:-1]
        elif index == len(resultados) - 1:  # Último segmento
            gx_completo[start_row + 1:start_row + segmento_altura] = gx[1:]
            gy_completo[start_row + 1:start_row + segmento_altura] = gy[1:]
        else:  # Segmentos intermedios
            gx_completo[start_row + 1:start_row + segmento_altura - 1] = gx[1:-1]
            gy_completo[start_row + 1:start_row + segmento_altura - 1] = gy[1:-1]





def parallel_sobel_convolution(image, num_segments=None):
    """
    Aplica la convolución Sobel en paralelo.

    Args:
    - image: imagen de entrada.
    - num_segments: número de segmentos para procesamiento paralelo.

    Returns:
    - gradient: imagen resultante con la magnitud del gradiente.
    """
    # Número de segmentos predeterminado
    num_segments = num_segments or mp.cpu_count()
    
    # Dividir la imagen
    segments = split_image(image, num_segments)
    
    # Procesar segmentos en paralelo
    with mp.Pool(processes=num_segments) as pool:
        resultados = pool.map(sobel_convolution_segment, segments)
    
    # Inicializar matrices completas
    gx_completo = np.zeros_like(image, dtype=np.float32)
    gy_completo = np.zeros_like(image, dtype=np.float32)
    
    # Reconstruir la imagen
    reconstruir_imagen(resultados, gx_completo, gy_completo)
    
    # Calcular magnitud del gradiente
    gradient = np.sqrt(gx_completo**2 + gy_completo**2)
    return np.clip(gradient, 0, 255).astype(np.uint8)




def main():
    # Cargar imagen en escala de grises
    image = cv2.imread('BordesImagen/image2.png', cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: No se pudo cargar la imagen.")
        exit(1)

    # Desenfocar la imagen
    image_blurred = cv2.blur(image, (3, 3))

    # Probar diferentes números de segmentos
    segment_options = [2, 4, 8, 16]
    for num_segments in segment_options:
        start_time = time.time()
        result = parallel_sobel_convolution(image_blurred, num_segments)
        elapsed_time = time.time() - start_time

        # Guardar resultado
        output_filename = f'resultados/parallel_blurred_sobel_{num_segments}_segments.jpg'
        cv2.imwrite(output_filename, result)
        
        print(f"Segmentos: {num_segments}, Tiempo: {elapsed_time:.4f} segundos")

if __name__ == '__main__':
    main()
