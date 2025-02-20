import numpy as np, os, zipfile
from settings import newModel, model

def data():

    if newModel:
        
        # Проверка на существование файла и его удаление, если файл есть
        if os.path.exists(model):
            os.remove(model)
        
        # Первый фильтр
        filters1 = np.random.randn(2, 5, 5) / np.sqrt(5)
        fOffset1 = np.random.randn(2)
        # (канал,фильтр,второй фильтр,размерность фильтра x,размерность фильтра y)
        filters2 = np.random.randn(2, 2, 2, 3, 3)
        # (канал,фильтр)
        fOffset2 = np.random.randn(2, 2)
        #Входной слой весов (делим на корень чтобы умекньшить сгенерированые веса (хз gpt порекомендовал))
        # inputWeights = np.random.randn(784, 784)/ np.sqrt(784)
        # weightsLayer_1 = np.random.randn(784, 128)/ np.sqrt(784)
        # weightsLayer_2 = np.random.randn(128, 64)/ np.sqrt(128)
        # weightsLayer_3 = np.random.randn(64, 10)/ np.sqrt(64)

        #Слои Смещений 
        # offset_1 = np.random.randn(784)
        # offset_2 = np.random.randn(128)
        # offset_3 = np.random.randn(64)
        offsetOutput = np.random.randn(10)

        # Сохраняем все массивы в одном файле
        np.savez_compressed(model,
        filters1 = filters1, 
        filters2 = filters2,
        fOffset1 = fOffset1,     
        fOffset2 = fOffset2,       
        # inputWeights = inputWeights,
        # weightsLayer_1 = weightsLayer_1, 
        # weightsLayer_2 = weightsLayer_2,
        # weightsLayer_3 = weightsLayer_3,
        # offset_1 = offset_1,
        # offset_2 = offset_2,
        # offset_3 = offset_3,
        offsetOutput = offsetOutput
        )
        return np.load(model)
    else:
        # Проверяем, существует ли файл
        if not os.path.exists(model):
            print("Модель не найдена!")
        else:   
            try:
                with zipfile.ZipFile(model, 'r') as zip_ref:
                    zip_ref.testzip()  # Проверяет целостность архива
            except zipfile.BadZipFile:
                print(f"Файл model.npz поврежден") 
            return np.load(model)

def dataGen():
    print('')


