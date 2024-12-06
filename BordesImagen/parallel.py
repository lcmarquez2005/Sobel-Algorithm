import numpy as np
import cv2
import time
import multiprocessing as mp

# Definir máscaras Sobel
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

def sobel_convolution_segment(args):

    segment, start_row,end_row = args #atributos necesarios para que en el metodo main la imagen pueda ser mapeada
    height, width = segment.shape
    
    gx = np.zeros_like(segment, dtype=np.float32) #grandiente de sobel en X
    gy = np.zeros_like(segment, dtype=np.float32) #grandiente de sobel en Y

    # Aplicar Sobel en cada píxel del segmento
    for i in range(1, height - 1): #vamos a iterar los pixeles de la altura del segmento que nos pasaron excepto 1 (borde)
        for j in range(1, width - 1): #vamos a iterar los pixeles de el ancho del segmento que nos pasaron excepto 1 (borde)
            region = segment[i - 1:i + 2, j - 1:j + 2]
            gx[i, j] = np.sum(region * sobel_x)
            gy[i, j] = np.sum(region * sobel_y)

    return gx, gy, start_row





def split_image(image, num_segments):
    """Metodo para dividir la imagen en segmentos que seran los procesados"""
    height, width = image.shape
    segment_height = height // num_segments
    segments = []

    for i in range(num_segments):
        start_row = max(0, i * segment_height - 1)  # Division superior
        end_row = min(height, (i + 1) * segment_height + 1)  # Division inferior
        segment = image[start_row:end_row, :]
        segments.append((segment, start_row, end_row))
    
    return segments



def reconstruir_imagen(resultados, gx_completo, gy_completo):
    """Metodo para reconstruir la imagen"""
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





def parallel_sobel(image, num_segments=None):

    # Número de segmentos predeterminado, 
    num_segments = num_segments or mp.cpu_count() ##DETECTAMOS los nucleos del procesador automaticamente si no se pasa un valor
    
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




# Metodo principal que sirve para poder generar la pool de procesos mediante Multiprocessing (python libreria)
def main():
    semgmentos = 8  or mp.cpu_count()
    # Cargar imagen en escala de grises
    image = cv2.imread('BordesImagen/image2.png', cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("error, no encuentro la imagen")
        exit(1)

    # Desenfocar la imagen, para eliminar un poco del ruido
    image_blurred = cv2.blur(image, (3, 3))

    #empezamos a tomar el tiempo de ejecucion, para medir desde que mandamos a llamr al metodo de sobel 
    start_time = time.time()
    result = parallel_sobel(image_blurred, semgmentos)
    elapsed_time = time.time() - start_time

    # Guardar resultado
    output_filename = f'resultados/parallel_sobel_{semgmentos}_segmentos.jpg'
    cv2.imwrite(output_filename, result)
    
    print(f"Segmentos: {semgmentos}, Tiempo: {elapsed_time:.4f} segundos")



# comprobacion de que estemos en el directorio al momennto de ejecutar. 
# Puede no ser necesario si las imagenes son cargadas del mismo sitio del que las vas a ejecutar
if __name__ == '__main__':
    main()
