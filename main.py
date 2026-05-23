import cv2
import numpy as np
from skimage import segmentation, color
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve

def colorize_bw_optimization(image_path: str, n_segments: int = 500) -> np.ndarray:
    """
    Цветизация черно-белого изображения методом оптимизации (Gal et al., 2015).
    """
    # 1. Загрузка и приведение к градациям серого
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    gray_3ch = np.stack([gray] * 3, axis=-1)
    
    # 2. Преобразование в L*a*b* (L сохраняется из оригинала)
    lab = color.rgb2lab(gray_3ch)
    L = lab[:, :, 0].copy()
    
    # 3. Суперпиксельная сегментация
    labels = labels.astype(np.int32)
    unique_labels = np.unique(labels)
    mapping = np.zeros(unique_labels.max() + 1, dtype=np.int32)
    mapping[unique_labels] = np.arange(len(unique_labels))
    labels = mapping[labels]
    num_labels = len(unique_labels)
    
    # 4. Инициализация цветовых подсказок (a* и b*)
    sp_means_a = np.zeros(num_labels)
    sp_means_b = np.zeros(num_labels)

    np.random.seed(42)
    hint_indices = np.random.choice(num_labels, size=int(num_labels * 0.1), replace=False)
    sp_means_a[hint_indices] = np.random.uniform(-50, 50, size=len(hint_indices))
    sp_means_b[hint_indices] = np.random.uniform(-50, 50, size=len(hint_indices))
    
    # 5. Построение графа смежности суперпикселей
    rows, cols, weights = [], [], []
    kernel = np.ones((3, 3), dtype=np.uint8)
    
    for i in range(num_labels):
        mask_i = (labels == i)
        dilated = cv2.dilate(mask_i.astype(np.uint8), kernel).astype(bool)
        dilated[mask_i] = False  # исключаем сам суперпиксель
        
        neighbor_labels = np.unique(labels[dilated])
        for j in neighbor_labels:
            if j > i:  # избегаем дублирования пар
                edge_mask = mask_i | dilated
                grad_mean = gray[edge_mask].std() + 1e-6  # защита от деления на 0
                w = 1.0 / grad_mean
                
                rows.extend([i, j])
                cols.extend([j, i])
                weights.extend([w, w])
    
    W = csr_matrix((weights, (rows, cols)), shape=(num_labels, num_labels))
    D = diags(np.ones(num_labels), 0)  # вес "подсказки" цвета
    
    # 6. Решение оптимизационной задачи для a* и b* отдельно
    def solve_channel(hints):
        A = D + W
        # Используем разреженный решатель для эффективности
        return spsolve(A.tocsr(), hints)
    
    opt_a = solve_channel(sp_means_a)
    opt_b = solve_channel(sp_means_b)
    
    # 7. Восстановление каналов на пиксельном уровне
    out_a = np.zeros_like(L)
    out_b = np.zeros_like(L)
    for i in range(num_labels):
        mask = (labels == i)
        out_a[mask] = opt_a[i]
        out_b[mask] = opt_b[i]
    
    # 8. Обратное преобразование в RGB
    out_lab = np.stack([L, out_a, out_b], axis=-1)
    out_rgb = color.lab2rgb(out_lab)
    
    return (out_rgb * 255).astype(np.uint8)

# Пример использования
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    IMAGE_PATH = "bw_photo.jpg"  # Замените на путь к вашему ЧБ изображению
    
    try:
        colorized = colorize_bw_optimization(IMAGE_PATH, n_segments=400)
        original = cv2.imread(IMAGE_PATH)
        # OpenCV загружает в BGR, matplotlib ожидает RGB
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(original_rgb)
        plt.title("Original B&W")
        plt.axis("off")
        
        plt.subplot(1, 2, 2)
        plt.imshow(colorized)
        plt.title("Optimized Colorization (Gal et al., 2015)")
        plt.axis("off")
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Ошибка: {e}")