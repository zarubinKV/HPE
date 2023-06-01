# HPE

## Подготовка

1) pip install -r requirements.txt
2) Создать папку data
3) Скачать веса в созданную папку data\
3.1. Наша модель: [Weight](https://drive.google.com/drive/folders/1E0l6ZtNgU3nuXZxIfEJQDX3SF79RnuQc?usp=sharing)\
3.2. Open source модель: [Weight](https://github.com/OlgaChernytska/2D-Hand-Pose-Estimation-RGB/blob/972c2102d95e14ebb37b1cbd452018ebd6706a44/weights/model_final)

### Датасет
[FreiHAND Dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html)

## Наша модель

Ноутбук с обучением и метриками обученной модели /Notebooks/HPE.ipynb

Модель "mobilenetv2_050". Полученная метрика на тестовых данных:

*Test OKS: 0.9984014211599824*

Время инференса батча из 48 изображений: 0,015 секунд

## Open source модель

[GitHub](https://github.com/OlgaChernytska/2D-Hand-Pose-Estimation-RGB/blob/main/utils/model.py)

Ноутбук с метриками обученной модели /Notebooks/HPE_OS.ipynb

*Test OKS: 0.9983032275185367*

Время инференса одного изображения: 0,3 секунды

## Рисование на экране

Решено использовать нашу модель, так как оно значительно быстрее.

Также в реализации используется предобученная [YOLOv4-Tiny](https://github.com/cansik/yolo-hand-detection). Её применение обусловлено особенностями датасета, на котором происходило обучение: на всех изображениях рука находится в центре кадра, поэтому с помощью yolo мы определяем, где на кадре находится рука, чтобы решить проблему привязки "поиска руки" к центру изображения.

Итоговый результат рисования:
