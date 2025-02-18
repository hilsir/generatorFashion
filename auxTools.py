import struct, os, numpy as np


#Для открытия файлов с изображениями
def openFileImage(filename):
    # Проверка, существует ли файл
    if not os.path.isfile(filename):
      print(f"Файл {filename} не найден или это директория.")
    else:
        with open(filename, 'rb') as f:
            # Чтение заголовка 
            magic_number, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
            # Чтение данных изображений в массив
            images = np.frombuffer(f.read(), dtype=np.uint8)
            # Преобразование данных в форму 
            images = images.reshape(num_images, rows, cols)
            return images

#Для открытия файлов с масивами ответов
def openFileAnswers(filename):
# Открываем файл в бинарном режиме
    with open(filename, 'rb') as f:
        # Читаем заголовок (первые 8 байт содержат информацию о формате)
        magic_number, num_labels = struct.unpack('>II', f.read(8))
        
        # Проверяем правильность заголовка (magic_number должен быть 2049 для меток)
        if magic_number != 2049:
            raise ValueError(f"Неверный magic_number: {magic_number}, ожидался 2049")
        
        # Считываем метки (каждая метка - это 1 байт)
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels
