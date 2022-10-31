## Обучение Егора применению методов машинного обучения
- [x] Нормировать данные (min/max scaling, standard scaling)
- [x] Написать функцию для подсчёта функции потерь
- [x] Написать шаг оптимизации параметров MSE + градиентный спуск.
- [ ] Написать визуализацию результатов обучения (график лосса)
- [ ] Проверить результаты на тестовой выборке
  - [ ] Использовать ту же самую предобработку что и для train !!! (standard scaling)
  - [ ] Посчитать MSE на тесте
- [ ] Добавить больше признаков и обучить модель ещё раз
Подумать, что делать с не числовыми признаками? Как их интерпретировать?
binary encoding, OHE one hot encoding e.t.c.


- [x] Разобраться с train / val / test разбиением обучающей выборки.
- [x]  Обучить модель и нарисовать график значения функции ошибки  на train и val датасетах
- [x] Перенести код для обработки данных и обучения в отдельные py файлы, а в ноутбуке оставить лишь код для их вызова (т.е. условно есть функция train_one_step и она в файле, а в ноутбуке ты вызываешь её 300 раз
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
