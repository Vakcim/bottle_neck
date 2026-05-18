# T-Invest Trading Bot v0.1 — data layer

Первая версия проекта: загрузка инструментов и исторических свечей через T-Invest API, сохранение в Parquet/DuckDB, подготовка основы для будущих новостных признаков и ML.

> Важно: эта версия ничего не покупает и не продаёт. Это только слой данных.

## Что уже есть

- Подключение к T-Invest API через официальный Python SDK `tinkoff-investments`
- Загрузка исторических свечей по списку тикеров
- Автоматический поиск FIGI по тикеру
- Сохранение свечей в `data/candles/*.parquet`
- Локальная аналитика через DuckDB
- Конфиги для активов, настроек и риск-лимитов
- Заготовка для GDELT-новостей
- CLI-скрипты

## Структура

```text
tinvest_trading_bot_v0_1/
  config/
    assets.yaml
    risk.yaml
    settings.yaml
  data/
    candles/
    news/
    instruments/
  scripts/
    download_history.py
    inspect_storage.py
  src/
    connectors/
      tinkoff_client.py
      gdelt_client.py
    data/
      storage.py
    features/
      market_features.py
    settings.py
  .env.example
  pyproject.toml
```

## Установка

```bash
cd tinvest_trading_bot_v0_1

python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# или
.venv\Scripts\activate     # Windows

pip install -e .
```

## Настройка токена

Создай файл `.env`:

```bash
cp .env.example .env
```

Вставь токен:

```env
TINVEST_TOKEN=твой_токен
TINVEST_SANDBOX=true
```

Для первого этапа sandbox не критичен, потому что мы только читаем рыночные данные, но настройка уже заложена.

## Настройка тикеров

Открой `config/assets.yaml` и измени список:

```yaml
assets:
  - ticker: SBER
    class_code: TQBR
  - ticker: GAZP
    class_code: TQBR
  - ticker: LKOH
    class_code: TQBR
```

## Скачать исторические свечи

Пример: дневные свечи за 2023–2025 годы.

```bash
python scripts/download_history.py \
  --from-date 2023-01-01 \
  --to-date 2025-12-31 \
  --interval day
```

Для часовых свечей:

```bash
python scripts/download_history.py \
  --from-date 2024-01-01 \
  --to-date 2025-12-31 \
  --interval hour
```

Для 15-минутных:

```bash
python scripts/download_history.py \
  --from-date 2025-01-01 \
  --to-date 2025-12-31 \
  --interval 15min
```

## Проверить хранилище

```bash
python scripts/inspect_storage.py
```

## Что дальше

Следующий этап:

1. добавить полноценную загрузку новостей;
2. сопоставить новости с тикерами;
3. посчитать рыночные признаки;
4. собрать первый датасет;
5. написать бэктестер.

## Ограничения v0.1

- Нет торговли.
- Нет бэктестера.
- Нет ML-модели.
- Нет гарантии, что все тикеры доступны по выбранному классу торгов.
- Исторические лимиты T-Invest API могут отличаться по интервалам свечей.
