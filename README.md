# Предсказание отклика на банковскую маркетинговую кампанию

## Описание задачи

Модель ранжирует клиентов по вероятности отклика `y` до звонка. Данные несбалансированы, а число контактов ограничено.

## Цель проекта

Повысить recall положительных откликов в верхней доле списка обзвона. Основная метрика — PR-AUC; продуктовый срез — recall при бюджете 15% клиентов.

## Архитектура решения

Строгая схема → сохранение исходного хронологического порядка → первые 60% train, следующие 20% validation, последние 20% test → preprocessing внутри pipeline → Dummy, Logistic Regression, class-weighted Logistic Regression, Random Forest, Gradient Boosting и калиброванный Random Forest. Модель выбирается на validation; для каждого batch стабильно выбираются ровно верхние `ceil(N × budget_fraction)` строк, а `duration` всегда удалён как post-contact leakage. Перед scoring проверяются версия bundle, точный порядок признаков, бюджет и интерфейс модели.

## Структура каталогов

`src/bank_marketing` — данные, модели, генератор и CLI; `tests` — автономный workflow; `data` — контракт; `artifacts`/`reports` создаются локально; `.github/workflows` — CI.

## Используемые технологии

Python 3.11, NumPy, pandas, scikit-learn, joblib, pytest, Ruff.

## Требования к окружению

Python 3.11.15; версии библиотек зафиксированы в `pyproject.toml`.

## Установка

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Подготовка данных

Источник: [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing). Лицензию нужно проверить на официальной карточке; raw-файлы не коммитятся. Не меняйте порядок `bank-additional-full.csv`. Для smoke-run: `make smoke`.

## Запуск обучения

```bash
bank-train --data data/smoke.csv --artifact artifacts/model.joblib --report reports/validation_metrics.json
```

## Запуск оценки

```bash
bank-evaluate --data data/smoke.csv --artifact artifacts/model.joblib --metrics reports/test_metrics.json --errors reports/test_errors.csv
```

## Запуск инференса

```bash
bank-predict --data data/smoke.csv --artifact artifacts/model.joblib --output reports/predictions.csv
```

## Метрики

PR-AUC, ROC-AUC, Brier score, precision, recall, F1, confusion matrix и фактическое число выбранных клиентов. Операционная политика — ровно верхние 15% каждого batch с устойчивым tie-break по порядку строк; cutoff сохраняется для аудита. Все денежные эффекты потребовали бы отдельных модельных предположений.

## Тестирование

`make check` запускает Ruff и pytest без скачивания данных.

## Ограничения

Порядок строк — лишь приближение ко времени: в публичном наборе нет полной даты каждого контакта. Калибровка и порог могут деградировать при смене кампании. Модель не определяет причинный эффект звонка.

## Полученные результаты

Реальный UCI-набор централизованно ещё не оценён; метрики не заявляются. Smoke-результаты подтверждают только работоспособность и не будут представлены как качество модели.

## Статус проекта

Код и проверки готовы; фиксация версии UCI-данных и итоговый test-отчёт ожидают централизованного запуска.
