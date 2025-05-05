<h1 align="center">
  Speaker Verification System for Emotional Voice<br>
  Голосовая верификация в условиях эмоционального голоса
</h1>


<p align="center">
  <a href="#about">Описание</a> •
  <a href="#installation">Установка</a> •
  <a href="#how-to-use">Загрузка данных</a> •
  <a href="#examples">Обучение моделей</a> •
  <a href="#credits">Тестирование</a> •
  <a href="#license">Благодарности</a> •
  <a href="#license">Лицензия</a>
</p>

## Описание

Данный репозиторий содержит фреймворк для дообучения и оценки моделей, решающих задачу голосовой верификации в условиях эмоционального голоса. 

В проекте рассмотрены две архитектуры:

- ECAPA-TDNN
- WavLM-Large + ECAPA-TDNN (ECAPA-TDNN решает задачу верификации, используя признаки из WavLM-Large)

Репозиторий содержит скрипты для:

- Предобработки и организации аудиоданных;
- Добавления шумовых аугментаций в процессе обучения;
- Дообучения моделей ECAPA-TDNN и WavLM-Large + ECAPA-TDNN на эмоциональных данных;
- Дообучения с использованием метода обратного градиента (Gradient Reversal Layer, GRL) для повышения инвариантности к эмоциям;
- Оценки качества верификации с использованием метрики равной ошибки (Equal Error Rate, EER).

## Установка

Следуйте следующим шагам:

0. (Опционально) Создайте и активируйте новое виртуальное окружение с помощью [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html):

   ```bash
   # создать окружение
   conda create -n project_env python=3.9

   # активировать окружение 
   conda activate project_env
   ```

1. Клонируйте репозиторий:

   ```bash
   git clone <URL-репозитория>
   cd SVS_for_Emotional_Voice
   ```

2. Установите зависимости:

   ```bash
   ./install.sh
   ```

## Загрузка данных

Чтобы загрузить набор данных CREMA-D для обучения и тестирования, запустите следующую команду:

```bash
python3 scripts/download_cremad.py
```

Чтобы загрузить набор данных VoxCeleb1 для обучения и тестирования, запустите команду:

```bash
python3 scripts/download_voxceleb.py
```

Чтобы загрузить веса предобученных моделей и состояния моделей после дообучения, запустите команду:

```bash
python3 scripts/download_weights.py
```


## Обучение
### ECAPA-TDNN
Чтобы начать дообучение ECAPA-TDNN на CREMA-D, запустите команду:

```bash
python3 train.py -cn=ecapatdnn_finetune
```

Чтобы начать дообучение ECAPA-TDNN на CREMA-D и VoxCeleb1, запустите команду:

```bash
python3 train.py -cn=ecapatdnn_finetune datasets=big_train_dataset loss_function.n_speakers=117 
```

Чтобы начать дообучение ECAPA-TDNN c использованием GRL на CREMA-D, запустите команду:

```bash
python3 train.py -cn=ecapatdnn_grl 
```

Чтобы начать дообучение ECAPA-TDNN c использованием GRL на CREMA-D и VoxCeleb1, запустите команду:

```bash
python3 train.py -cn=ecapatdnn_grl datasets=big_multitask_dataset loss_function.n_speakers=117 
```
### WavLM-Large + ECAPA-TDNN
Чтобы начать дообучение WavLM-Large + ECAPA-TDNN на CREMA-D, запустите команду:

```bash
python3 train.py -cn=wavlm_finetune
```
Чтобы начать дообучение WavLM-Large + ECAPA-TDNN на CREMA-D и VoxCeleb1, запустите команду:

```bash
python3 train.py -cn=wavlm_finetune datasets=big_train_dataset loss_function.n_speakers=117 
```

Чтобы начать дообучение WavLM-Large + ECAPA-TDNN c использованием GRL на CREMA-D, запустите команду:

```bash
python3 train.py -cn=wavlm_grl 
```
Чтобы начать дообучение WavLM-Large + ECAPA-TDNN c использованием GRL на CREMA-D и VoxCeleb1, запустите команду:

```bash
python3 train.py -cn=wavlm_grl datasets=big_multitask_dataset loss_function.n_speakers=117 
```

## Тестирование
Для тестирования модели, запустите команду:

```bash
python3 inference.py \
        model=MODEL_NAME \
        inferencer.from_pretrained=PATH_TO_WEIGHTS_OF_MODEL \
        datasets.test.list_path=PATH_TO_TEST_PAIRS \
        datasets.test.data_path=PATH_TO_TEST_DATA
```
Где 'MODEL_NAME' - название модели ('ecapa_tdnn' или 'wavlm_large'), 'PATH_TO_WEIGHTS_OF_MODEL' - путь к файлу с весами модели, 'PATH_TO_TEST_PAIRS' - путь к списку тестовых пар, 'PATH_TO_TEST_DATA' - путь к папке с тестовыми аудиозаписями.

### Примеры тестирования

1. Тестирование предобученной модели ECAPA-TDNN:

    ```bash
    python3 inference.py \
            model=ecapa_tdnn \
            inferencer.from_pretrained='data/weights/ecapatdnn_pretrained.model'        # Тестирование на CREMA-D        
    ```

    Результаты
    - EER на CREMA-D: 13.92%
    - EER на Vox1_O (VoxCeleb1): 0.96%
    - EER на RAVDESS: 42.30%

2. Лучший результат дообученной модели ECAPA-TDNN:

    ```bash
    python3 inference.py \
            model=ecapa_tdnn \
            inferencer.from_pretrained='data/weights/ecapatdnn_weights.pth'             # Тестирование на CREMA-D
    ```

    Результаты
    - EER на CREMA-D: 13.46%
    - EER на Vox1_O (VoxCeleb1): 4.22%
    - EER на RAVDESS: 31.25%

3. Тестирование предобученной модели WavLM-Large + ECAPA-TDNN:

    ```bash
    python3 inference.py \
            model=wavlm_large \
            inferencer.from_pretrained='data/weights/wavlm_pretrained.pth'              # Тестирование на CREMA-D        
    ```

    Результаты
    - EER на CREMA-D: 12.04%
    - EER на Vox1_O (VoxCeleb1): 0.6%
    - EER на RAVDESS: 29.2%

4. Лучший результат после дообучения модели WavLM-Large + ECAPA-TDNN c использованием GRL:

    ```bash
    python3 inference.py \
            model=wavlm_large \
            inferencer.from_pretrained='pretrained/wavlm_weights.pth'                # Тестирование на CREMA-D 
    ```

    Результаты
    - EER на CREMA-D: 10.92%
    - EER на Vox1_O (VoxCeleb1): 1.9%
    - EER на RAVDESS: 28.17%

## Благодарности

Этот репозиторий основан на [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template). Реализация модели ECAPA-TDNN взята из работы [ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN/tree/main). Архитектура модели WavLM-Large + ECAPA-TDNN адаптирована из проекта [UniSpeech](https://github.com/microsoft/UniSpeech/tree/main)

## Лицензия

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
