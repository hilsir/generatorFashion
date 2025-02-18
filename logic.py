import numpy as np

def normalize(x):
    x_min = np.min(x)
    x_max = np.max(x)
    
    return (x - x_min) / (x_max - x_min)

def normalizeRes(x):
    # Для числовой стабильности вычитаем максимальное значение
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)  # Это предотвращает переполнение
    normalized = exp_x / np.sum(exp_x)  # Нормализуем и делим на сумму экспонент
    return np.round(normalized, 3)  # Округляем до двух знаков после запятой

def relu(x):
    return np.maximum(0, x)

def reluDerivative(x):
    return np.where(x > 0, 1, 0)

# умножаем слой неронов на веса следующего слоя + смещение
def forwardPass(neuronsLayer, weightsLayer, offset):
    return relu(np.dot(weightsLayer.T,neuronsLayer) + offset)  

# Перемножаем все вектора с нейронами
def error(errorOutput, neuronlayer, weightsLayer):
    return np.dot(weightsLayer,errorOutput.T) * reluDerivative(neuronlayer)

def backWeights(weightsLayer,neuronlayer,errorlayer,learningRate):
    return weightsLayer - np.outer(neuronlayer, errorlayer) * learningRate

def backNeuron(neuronlayer,errorlayer,learningRate):
    return neuronlayer - (errorlayer * learningRate)

def updateOffset(offset, error, learningRate):
    return offset - learningRate * np.sum(error)

# Для генератора________________________________________________________________________________________________ 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Применение фильтров
# stride шаг цикла. padding увеличить массив по краям
def convolution(image, filter, offset, stride=1, padding=0):
    
    # Добавляем паддинг к изображению
    if padding > 0:
        image = np.pad(image, padding,'constant', 0)
    
    #Размер выходного изображения
    imageHeight, imageWidth = image.shape
    filterHeight, filterWidth = filter.shape
    outputHeight = (imageHeight - filterHeight) // stride + 1
    outputWidth = (imageWidth - filterWidth) // stride + 1
    output = np.zeros((outputHeight, outputWidth))
    
    for i in range(outputHeight):
        for j in range(outputWidth):
            # Область изображения, к которой применяется фильтр
            region = image[i * stride:i * stride + filterHeight, j * stride:j * stride + filterWidth]
            output[i, j] = relu(np.sum(region * filter) + offset)
    return output

def maxPooling(image, poolSize, stride):

    #Размер выходного изображения
    imageHeight, imageWidth  = image.shape
    outputHeight = (imageHeight - poolSize) // stride + 1
    outputWidth = (imageWidth - poolSize) // stride + 1
    output = np.zeros((outputHeight, outputWidth))
    
    # Проходим по каждому блоку (окну) входного массива
    for h in range(outputHeight):
        for w in range(outputWidth):
            # Вычисляем границы региона
            startH = h * stride
            startW = w * stride
            endH = startH + poolSize
            endW = startW + poolSize
            
            # Берем максимальное значение из текущего региона
            region = image[startH:endH, startW:endW]
            output[h, w] = np.max(region)
            
    return output
