# PollenNER - Анализ медицинских текстов

PollenNER - это система для извлечения именованных сущностей и отношений из медицинских текстов, связанных с аллергией и пыльцой. Система использует современные методы глубокого обучения для распознавания:

- Топонимов (места)
- Лекарств
- Симптомов
- Аллергенов
- Частей тела
- Отношений между ними (например, "глаза has_symptom слезятся")

## Развертывание на HuggingFace Spaces

1. Создайте новый Space на HuggingFace:
   - Перейдите на [HuggingFace Spaces](https://huggingface.co/spaces)
   - Нажмите "Create new Space"
   - Выберите "Gradio" как SDK
   - Укажите имя для вашего Space (например, "pollen-ner")
   - Выберите "Public" для видимости

2. Загрузите файлы в Space:
   - `app.py` - основной файл с Gradio интерфейсом
   - `requirements.txt` - файл с зависимостями
   - `README.md` - документация проекта

3. Убедитесь, что ваши модели загружены на HuggingFace Hub:
   - NER модель: `pollen-ner-1500`
   - RE модель: `pollen-re-model`

4. Настройте переменные окружения в Space:
   - Перейдите в Settings -> Repository Secrets
   - Добавьте необходимые токены (если требуется)

## Использование

1. Введите текст в текстовое поле
2. Нажмите "Submit" или Enter
3. Получите результаты анализа:
   - Список найденных сущностей по категориям
   - Отношения между сущностями

## Примеры

```
Входной текст:
"В Московской области у меня началась аллергия на пыльцу березы, потекли глаза, нос, принимаю Зиртек и Назонекс."

Результаты:
Найденные сущности:
TOPONYM:
- Московской области

ALLERGEN:
- пыльцу березы

SYMPTOM:
- потекли глаза
- нос

MEDICINE:
- Зиртек
- Назонекс

Найденные отношения:
- глаза has_symptom потекли
- нос has_symptom течет
```

## Технические детали

- Модель NER основана на ruBERT с адаптером LoRA
- Модель RE использует классификацию отношений
- Интерфейс построен с использованием Gradio
- Поддерживается обработка длинных текстов с разбиением на предложения

## Требования

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.51.3+
- Gradio 4.0+
- PEFT 0.7.0+

## Лицензия

MIT License 