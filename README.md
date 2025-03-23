# ArticlesSimilarities

Finds similar articles among webpages using pretrained models for Russian and English. 
Check https://github.com/RaRe-Technologies/gensim-data

## Installation

Steps:

1. Please, use python 3.9 or higher
2. Install dependencies using requirements.txt
3. Download NLTK resources via Python cmd:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
```

4. config.yml:

* url_list - path to url list
* language - either 'ru' or 'en'
* min_sent_len - minimum sentence len to be processed
* n_closest - n closest articles to current article
		
5. run main.py

6. Test result is in text file result.txt using ./data/football_vs_movies.txt file as input in config file

Полный алгоритм
1. Калибровка камеры
Используйте шахматную доску для калибровки камеры. Этот шаг выполняется один раз, если параметры камеры не меняются.

python
Copy
import cv2
import numpy as np
import glob

# Параметры шахматной доски
squares_x = 10  # Количество квадратов по горизонтали
squares_y = 7   # Количество квадратов по вертикали
chessboard_size = (squares_x - 1, squares_y - 1)  # Количество углов
square_size = 30  # Размер квадрата в мм

# Подготовка 3D точек шахматной доски
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Массивы для хранения 3D точек и 2D точек на изображениях
objpoints = []  # 3D точки в реальном мире
imgpoints = []  # 2D точки на изображениях

# Загрузка изображений
image_folder = 'path/to/images/*.jpg'  # Укажите путь к папке с изображениями
images = glob.glob(image_folder)

# Критерии для уточнения углов
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Поиск углов шахматной доски на изображениях
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Поиск углов
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Уточнение найденных углов
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Визуализация углов
        cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        cv2.imshow('Chessboard', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Калибровка камеры
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

if ret:
    print("Матрица камеры:\n", mtx)
    print("Коэффициенты искажения:\n", dist)
else:
    print("Ошибка калибровки. Убедитесь, что шахматная доска найдена на всех изображениях.")
    exit()
2. Устранение искажений на изображении конвейерной ленты
После калибровки камеры устраните искажения на изображении конвейерной ленты, не обрезая изображение.

python
Copy
# Загрузка изображения конвейерной ленты
image_path = 'path/to/conveyor_image.jpg'  # Укажите путь к изображению
img = cv2.imread(image_path)
h, w = img.shape[:2]

# Устранение искажений (без обрезки)
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
undistorted_img = cv2.undistort(img, mtx, dist, None, newcameramtx)
3. Поиск контуров всех объектов
Найдите контуры всех объектов (камней) на изображении.

python
Copy
# Преобразование изображения в оттенки серого
gray = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)

# Бинаризация изображения
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Поиск контуров
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
4. Вычисление реальных размеров объектов
Для каждого контура вычислите реальные размеры объекта.

python
Copy
# Масштабный коэффициент (пиксели в миллиметры)
scale_factor = square_size / np.linalg.norm(objp[0] - objp[1])  # На основе размера квадрата шахматной доски

# Обработка каждого контура
for i, cnt in enumerate(contours):
    # Приблизительный прямоугольник, описывающий контур
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Визуализация контура
    cv2.drawContours(undistorted_img, [box], 0, (0, 255, 0), 2)

    # Вычисление ширины и высоты в пикселях
    width_px = np.linalg.norm(box[0] - box[1])  # Ширина в пикселях
    height_px = np.linalg.norm(box[1] - box[2])  # Высота в пикселях

    # Перевод в реальные размеры
    real_width = width_px * scale_factor
    real_height = height_px * scale_factor

    # Вывод размеров
    print(f"Объект {i + 1}: Ширина = {real_width:.2f} мм, Высота = {real_height:.2f} мм")

    # Отображение размеров на изображении
    text = f"Obj {i + 1}: {real_width:.1f}x{real_height:.1f} mm"
    cv2.putText(undistorted_img, text, (int(box[0][0]), int(box[0][1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# Показать результат
cv2.imshow('Undistorted Image with Contours', undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
Пояснения к коду
Устранение искажений без обрезки:

Используется функция cv2.undistort, которая корректирует всё изображение, не обрезая его.

Масштабный коэффициент:

Используется для перевода размеров из пикселей в миллиметры. Он вычисляется на основе известного размера квадрата шахматной доски.

Контуры объектов:

Каждый контур аппроксимируется минимальным ограничивающим прямоугольником (cv2.minAreaRect), чтобы получить ширину и высоту объекта.

Реальные размеры:

Размеры в пикселях умножаются на масштабный коэффициент для получения реальных размеров в миллиметрах.

Визуализация:

На изображении отображаются контуры объектов и их реальные размеры.

Итог
Этот алгоритм позволяет:

Калибровать камеру с использованием шахматной доски.

Устранить искажения на изображении конвейерной ленты без обрезки.

Найти контуры всех объектов (камней) и вычислить их реальные размеры.
