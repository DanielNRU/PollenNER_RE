import gradio as gr
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import AutoModelForSequenceClassification
import re
from typing import List, Dict, Tuple

# Словари для маппинга
id2label = {
    0: "O",
    1: "B-TOPONYM",
    2: "I-TOPONYM",
    3: "B-MEDICINE",
    4: "I-MEDICINE",
    5: "B-SYMPTOM",
    6: "I-SYMPTOM",
    7: "B-ALLERGEN",
    8: "I-ALLERGEN",
    9: "B-BODY_PART",
    10: "I-BODY_PART"
}

label2id = {v: k for k, v in id2label.items()}

# Загрузка моделей
def load_models():
    """Загружает NER и RE модели из HuggingFace Hub"""
    # Загрузка NER модели
    ner_model = AutoModelForTokenClassification.from_pretrained(
        "DeepPavlov/rubert-base-cased",
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id
    )
    ner_model.load_adapter("DanielNRU/pollen-ner-1500")
    ner_tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
    
    # Загрузка RE модели
    re_model = AutoModelForSequenceClassification.from_pretrained("DanielNRU/pollen-re-model_")
    re_tokenizer = AutoTokenizer.from_pretrained("DanielNRU/pollen-re-model_")
    
    return ner_model, ner_tokenizer, re_model, re_tokenizer

# Загрузка моделей при старте
ner_model, ner_tokenizer, re_model, re_tokenizer = load_models()

relation_labels = ["no_relation", "has_symptom", "has_medicine"]

def predict_entities(text: str) -> List[Dict]:
    """Извлекает сущности из текста с помощью NER модели"""
    # Токенизация текста
    inputs = ner_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    # Получение предсказаний
    with torch.no_grad():
        outputs = ner_model(**inputs)
        predictions = outputs.logits.argmax(dim=-1)[0]
    
    # Преобразование токенов обратно в слова
    tokens = ner_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    # Сбор сущностей
    entities = []
    current_entity = None
    current_text = ""
    
    for token, pred in zip(tokens, predictions):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
            
        label = id2label[pred.item()]
        
        if label.startswith("B-"):
            if current_entity:
                entities.append({
                    "text": current_text.strip(),
                    "type": current_entity
                })
            current_entity = label[2:]
            current_text = token.replace("##", "")
        elif label.startswith("I-"):
            if current_entity:
                current_text += " " + token.replace("##", "")
        else:
            if current_entity:
                entities.append({
                    "text": current_text.strip(),
                    "type": current_entity
                })
                current_entity = None
                current_text = ""
    
    return entities

def insert_entity_markers(text: str, entities: List[Dict]) -> str:
    """Вставляет маркеры сущностей в текст"""
    # Сортируем сущности по позиции в тексте
    entities = sorted(entities, key=lambda x: text.find(x["text"]))
    
    # Вставляем маркеры
    offset = 0
    for entity in entities:
        start = text.find(entity["text"]) + offset
        end = start + len(entity["text"])
        
        text = text[:start] + f"[{entity['type']}]" + text[start:end] + f"[/{entity['type']}]" + text[end:]
        offset += len(f"[{entity['type']}]") + len(f"[/{entity['type']}]")
    
    return text

def process_text(text: str) -> Tuple[List[Dict], List[Dict]]:
    """Обрабатывает текст и возвращает сущности и отношения"""
    # Получаем сущности
    entities = predict_entities(text)
    
    # Группируем сущности по типам
    entities_by_type = {}
    for entity in entities:
        if entity["type"] not in entities_by_type:
            entities_by_type[entity["type"]] = []
        entities_by_type[entity["type"]].append(entity["text"])
    
    # Находим отношения
    relations = []
    for i, entity1 in enumerate(entities):
        for j, entity2 in enumerate(entities[i+1:], i+1):
            # Формируем текст для классификации отношений
            relation_text = f"{entity1['text']} [SEP] {entity2['text']}"
            
            # Получаем предсказание отношения
            inputs = re_tokenizer(relation_text, return_tensors="pt", truncation=True)
            with torch.no_grad():
                outputs = re_model(**inputs)
                prediction = outputs.logits.argmax(dim=-1).item()
            
            if prediction > 0:  # Если есть отношение (не no_relation)
                relations.append({
                    "source": entity1["text"],
                    "target": entity2["text"],
                    "type": relation_labels[prediction]
                })
    
    return entities_by_type, relations

def format_output(entities_by_type: Dict[str, List[str]], relations: List[Dict]) -> str:
    """Форматирует результаты для отображения"""
    output = "Найденные сущности:\n"
    
    # Выводим сущности по типам
    for entity_type, texts in entities_by_type.items():
        output += f"\n{entity_type}:\n"
        for text in texts:
            output += f"- {text}\n"
    
    # Выводим отношения
    if relations:
        output += "\nНайденные отношения:\n"
        for relation in relations:
            output += f"- {relation['source']} {relation['type']} {relation['target']}\n"
    
    return output

def analyze_text(text: str) -> str:
    """Основная функция для анализа текста"""
    entities_by_type, relations = process_text(text)
    return format_output(entities_by_type, relations)

# Создаем Gradio интерфейс
interface = gr.Interface(
    fn=analyze_text,
    inputs=gr.Textbox(
        lines=5,
        placeholder="Введите текст для анализа...",
        label="Текст"
    ),
    outputs=gr.Textbox(
        lines=10,
        label="Результаты анализа"
    ),
    title="PollenNER — извлечения сущностей для анализа сообщений Пыльца Club",
    description="""
    Система для извлечения именованных сущностей и отношений между ними из сообщений.
    Распознает топонимы, лекарства, симптомы и аллергены.
    """,
    examples=[
        ["В Московской области у меня началась аллергия на пыльцу березы, потекли глаза, нос, принимаю Зиртек и Назонекс."],
        ["В Санкт-Петербурге обострилась аллергия на пыльцу ольхи, чешутся глаза и горло, врач прописал Кларитин."]
    ]
)

# Запускаем интерфейс
if __name__ == "__main__":
    interface.launch() 