import numpy as np
import cv2
import time
import multiprocessing as mp

# Definir máscaras Sobel
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

def sobel_convolution_segment(args):
    """
    Función para realizar la convolución Sobel en un segmento de imagen.
    
    Parámetros:
    - args: tupla que contiene (segment, start_row, end_row)
        segment: porción de imagen a procesar
        start_row: fila de inicio del segmento
        end_row: fila final del segmento
    
    Retorna:
    - gx: gradiente en dirección x para el segmento
    - gy: gradiente en dirección y para el segmento
    """
    segment, start_row, end_row = args
    height, width = segment.shape
    
    gx = np.zeros_like(segment, dtype=np.float32)
    gy = np.zeros_like(segment, dtype=np.float32)

    # Realizar la convolución Sobel
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            region = segment[i - 1:i + 2, j - 1:j + 2]
            gx[i, j] = np.sum(region * sobel_x)
            gy[i, j] = np.sum(region * sobel_y)

    return gx, gy, start_row

def split_image(image, num_segments):
    """
    Divide la imagen en segmentos de forma simple.
    
    Args:
    - image: imagen de entrada
    - num_segments: número de segmentos
    
    Returns:
    - lista de tuplas (segmento, start_row, end_row)
    """
    height, width = image.shape
    segment_height = height // num_segments
    
    segments = []
    for i in range(num_segments):
        start_row = i * segment_height
        end_row = (i + 1) * segment_height if i < num_segments - 1 else height
        
        segment = image[start_row:end_row, :]
        segments.append((segment, start_row, end_row))
    
    return segments



def parallel_sobel_convolution(image, num_segments=None):
    """Aplicar convolución Sobel en paralelo."""
    # Usar número de núcleos de CPU por defecto
    num_segments = num_segments or mp.cpu_count()
    
    # Dividir imagen
    segments = split_image(image, num_segments)
    
    # Procesar segmentos en paralelo
    with mp.Pool(processes=num_segments) as pool:
        # Aplicar función Sobel a cada segmento
        results = pool.map(sobel_convolution_segment, segments)
    

    # Reconstruir imagen
    gx_full = np.zeros_like(image, dtype=np.float32)
    gy_full = np.zeros_like(image, dtype=np.float32)
    
    for gx, gy, start_row in results:
        segment_height = gx.shape[0]
        gx_full[start_row:start_row+segment_height, :] = gx
        gy_full[start_row:start_row+segment_height, :] = gy
    
    # Calcular magnitud del gradiente
    gradient = np.sqrt(gx_full**2 + gy_full**2)
    return np.clip(gradient, 0, 255).astype(np.uint8)



    

def main():
    # Leer imagen en escala de grises
    image = cv2.imread('BordesImagen/image.jpg', cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: No se pudo cargar la imagen.")
        exit(1)

    # Paso 1: Aplicar filtro de desenfoque (blur)
    image_blurred = cv2.blur(image, (3, 3))

    # Probar diferentes números de segmentos
    segment_options = [2, 4, 12, 20]
    
    for num_segments in segment_options:
        # Paso 2: Aplicar la convolución Sobel en paralelo
        start_time = time.time()
        result = parallel_sobel_convolution(image_blurred, num_segments)
        elapsed_time = time.time() - start_time

        # Guardar resultado
        output_filename = f'resultados/parallel_blurred_sobel_{num_segments}_segments.jpg'
        cv2.imwrite(output_filename, result)
        
        print(f"Segmentos: {num_segments}")
        print(f"Tiempo de procesamiento: {elapsed_time:.4f} segundos\n")




if __name__ == '__main__':
    main()