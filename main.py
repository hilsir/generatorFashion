import matplotlib
matplotlib.use('TkAgg') 
import numpy as np
import matplotlib.pyplot as plt 
from dateModels import data
from auxTools import openFileAnswers, openFileImage
from logic import forwardPass, error, backWeights, backNeuron, normalize, normalizeRes, updateOffset, convolution, sigmoid, maxPooling
from settings import model, training, era

if training:
    # для тренеровки
    # filenameIdx3 = 'mnistDataSet/numbers/train-images-idx3-ubyte'
    # filenameIdx1 = 'mnistDataSet/numbers/train-labels-idx1-ubyte'

    filenameIdx3 = 'mnistDataSet/fashion/train-images-idx3-ubyte'
    filenameIdx1 = 'mnistDataSet/fashion/train-labels-idx1-ubyte'
else:
    # для тестов
    # filenameIdx3 = 'mnistDataSet/numbers/t10k-images-idx3-ubyte'
    # filenameIdx1 = 'mnistDataSet/numbers/t10k-labels-idx1-ubyte'

    filenameIdx3 = 'mnistDataSet/fashion/t10k-images-idx3-ubyte'
    filenameIdx1 = 'mnistDataSet/fashion/t10k-labels-idx1-ubyte'
    era = 1

#Данные из датасета
images = openFileImage(filenameIdx3)
answers = openFileAnswers(filenameIdx1)

# Загрузка всех массивов
filters1 = data()['filters1']
fOffset1 = data()['fOffset1']

activationLayer1 = []
Pooling1 = []
# Применение первых фильтров
for f in range(filters1.shape[0]):
    activationLayer1.append(convolution(images[0],filters1[f],fOffset1[f],1))
# Пулинг
for f in range(filters1.shape[0]):
    Pooling1.append(maxPooling(activationLayer1[f],2,2))



# print(Pooling1[0])
plt.imshow(Pooling1[0], cmap='gray')  # cmap='gray' для черно-белого изображения

plt.show()

np.savez_compressed(model, 
        filters1 = filters1
        # offsetOutput = offsetOutput
        )