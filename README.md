## Обучение Егора применению методов машинного обучения

1. Удалить строки с нулевым бюджетом (проверить, что нет пропусков NaN, Null)
2. Нормировать данные (min/max scaling, standard scaling)
3. Написать функцию для подсчёта функции потерь
4. Написать шаг оптимизации параметров MSE + градиентный спуск.

Подумать, что делать с не числовыми признаками? Как их интерпретировать?
binary encoding, OHE one hot encoding e.t.c.

## Automatic Code Formatting
This repo uses `pre-commit` hooks to standardize code formatting and save mental energy.<br>
Install pre-commit package with:
```bash
pip install pre-commit
```
Next, install hooks from .pre-commit-config.yaml:
```bash
pre-commit install
```
After that your code will be automatically reformatted on every new commit.<br>

To format all files in the project use command:
```bash
pre-commit run -a
```
