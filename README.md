# DatasetViewer — Anomaly Detection for Oil Well Operations

Проект для обнаружения аномалий в данных нефтяных скважин на основе датасета Petrobras 3W.

## Цель

Обучить классификаторы, которые по временным рядам датчиков скважины определяют, какое нежелательное событие происходит (или всё нормально). Пять классов:

| Класс | Описание |
|-------|----------|
| 0 | Normal — штатная работа |
| 3 | DHSV Failure — отказ клапана |
| 4 | Severe Slugging — нестабильность потока |
| 7 | Scaling PCK — минеральные отложения |
| 9 | Hydrate — гидратная пробка |

## Датасет

**Petrobras 3W Dataset v2.0.0** (CC BY 4.0)
~895 parquet-файлов, ~22.5M строк, 5 сенсоров: `T-TPT`, `P-TPT`, `P-PDG`, `P-MON-CKP`, `T-JUS-CKP`
Папка: `petrobras 3W main dataset/`

## Стек

- Python 3.14, Jupyter Lab
- pandas, numpy, scipy
- scikit-learn, XGBoost
- matplotlib

## Структура

```
src/
  config.py          # константы: сенсоры, классы, размер окна
  loader.py          # загрузка parquet-файлов датасета
  preprocessor.py    # нормализация через StandardScaler
  features.py        # извлечение признаков по скользящему окну
  parquete_reading.py# утилита для просмотра parquet

notebooks/
  01_eda.ipynb       # EDA: загрузка, визуализация, нормализация
  02_features.ipynb  # инженерия признаков (121 696 окон × 31 признак)
  03_modeling.ipynb  # обучение и сравнение моделей

outputs/
  processed_data.parquet  # нормализованные данные
  features.parquet        # матрица признаков
  models/                 # обученные модели (.pkl)
  figures/                # графики
```

## Пайплайн

```
01_eda.ipynb → processed_data.parquet + scaler.pkl
      ↓
02_features.ipynb → features.parquet
      ↓
03_modeling.ipynb → random_forest.pkl / xgboost.pkl / logistic_regression.pkl
```

## Быстрый старт

```bash
# Активировать окружение
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # Linux/Mac

# Запустить Jupyter
jupyter lab

# Открыть и запустить ноутбуки по порядку: 01 → 02 → 03
```

## Ключевые параметры (config.py)

| Параметр | Значение | Смысл |
|----------|----------|-------|
| `WINDOW_SIZE` | 60 | длина временного окна |
| `WINDOW_STEP` | 30 | шаг окна (50% перекрытие) |
| `RANDOM_STATE` | 42 | фиксация случайности |

## Обученные модели (outputs/models/)

- `random_forest.pkl` — основная модель (24 MB)
- `xgboost.pkl` — альтернатива (649 KB)
- `logistic_regression.pkl` — базовая линия (1.4 KB)
- `scaler.pkl` — StandardScaler для нормализации входных данных
