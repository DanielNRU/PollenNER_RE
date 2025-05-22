import gradio as gr
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoModelForSequenceClassification
import re

# Маппинг классов NER
LABELS = ["TOPONYM", "MEDICINE", "SYMPTOM", "ALLERGEN", "BODY_PART"]
ID2LABEL = {
    0: "O",
    1: "B-TOPONYM", 2: "I-TOPONYM",
    3: "B-MEDICINE", 4: "I-MEDICINE",
    5: "B-SYMPTOM", 6: "I-SYMPTOM",
    7: "B-ALLERGEN", 8: "I-ALLERGEN",
    9: "B-BODY_PART", 10: "I-BODY_PART"
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# Маппинг классов RE
REL_LABELS = ["has_symptom", "has_medicine", "no_relation"]
REL_ID2LABEL = {i: l for i, l in enumerate(REL_LABELS)}
REL_LABEL2ID = {l: i for i, l in enumerate(REL_LABELS)}

# Загрузка моделей
def load_models():
    ner_model = AutoModelForTokenClassification.from_pretrained(
        "DanielNRU/pollen-ner",
        num_labels=len(ID2LABEL),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    ner_tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
    re_model = AutoModelForSequenceClassification.from_pretrained(
        "DanielNRU/pollen-re",
        num_labels=len(REL_LABELS),
        id2label=REL_ID2LABEL,
        label2id=REL_LABEL2ID
    )
    re_tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
    return ner_model, ner_tokenizer, re_model, re_tokenizer

ner_model, ner_tokenizer, re_model, re_tokenizer = load_models()

def split_sentences(text):
    # Примитивный сплиттер, можно заменить на nltk
    return [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]

def predict_entities(text):
    inputs = ner_tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=512)
    offset_mapping = inputs.pop('offset_mapping')[0]
    with torch.no_grad():
        outputs = ner_model(**inputs)
        predictions = outputs.logits.argmax(dim=-1)[0]
    entities = []
    current = None
    for idx, (pred, (start, end)) in enumerate(zip(predictions, offset_mapping)):
        start = int(start)
        end = int(end)
        if start == 0 and end == 0:
            continue
        label = ID2LABEL[pred.item()]
        if label.startswith('B-'):
            if current:
                entities.append(current)
            current = {'text': text[start:end], 'label': label[2:], 'start': start, 'end': end}
        elif label.startswith('I-') and current and label[2:] == current['label']:
            current['text'] += text[start:end]
            current['end'] = end
        else:
            if current:
                entities.append(current)
                current = None
    if current:
        entities.append(current)
    return entities

def insert_entity_markers(text, ent1, ent2):
    # Вставляет маркеры вокруг двух сущностей (как в обучении RE)
    if ent1['start'] < ent2['start']:
        first, second = ent1, ent2
    else:
        first, second = ent2, ent1
    first_tag = f"[{first['label']}]"
    first_end_tag = f"[/{first['label']}]"
    second_tag = f"[{second['label']}]"
    second_end_tag = f"[/{second['label']}]"
    text_marked = text[:second['start']] + second_tag + text[second['start']:second['end']] + second_end_tag + text[second['end']:]
    text_marked = text_marked[:first['start']] + first_tag + text_marked[first['start']:first['end']] + first_end_tag + text_marked[first['end']:]
    return text_marked

def predict_relations(text, entities):
    # Для каждой пары BODY_PART–SYMPTOM и BODY_PART–MEDICINE внутри предложения
    sentences = split_sentences(text)
    relations = []
    for sent in sentences:
        sent_start = text.find(sent)
        sent_end = sent_start + len(sent)
        ents_in_sent = [e for e in entities if e['start'] >= sent_start and e['end'] <= sent_end]
        for ent1 in ents_in_sent:
            for ent2 in ents_in_sent:
                if ent1 == ent2:
                    continue
                if ent1['label'] == 'BODY_PART' and ent2['label'] in ['SYMPTOM', 'MEDICINE']:
                    marked = insert_entity_markers(text, ent1, ent2)
                    inputs = re_tokenizer(marked, return_tensors='pt', truncation=True, max_length=256)
                    with torch.no_grad():
                        logits = re_model(**inputs).logits
                        pred = logits.argmax(-1).item()
                        rel_label = REL_ID2LABEL[pred]
                    if rel_label != 'no_relation':
                        relations.append({'head': ent1, 'tail': ent2, 'relation': rel_label})
    return relations

def analyze_text(text):
    entities = predict_entities(text)
    relations = predict_relations(text, entities)
    # Для HighlightedText: список (substring, label)
    highlights = []
    used_spans = set()
    for ent in entities:
        # Не добавлять дублирующиеся спаны
        span = (ent['start'], ent['end'], ent['label'])
        if span not in used_spans:
            highlights.append((ent['text'], ent['label']))
            used_spans.add(span)
    # Для отдельных полей
    toponyms = [ent['text'] for ent in entities if ent['label'] == 'TOPONYM']
    medicines = [ent['text'] for ent in entities if ent['label'] == 'MEDICINE']
    # Для симптомов — только из отношений has_symptom
    symptoms = []
    for rel in relations:
        if rel['relation'] == 'has_symptom' and rel['tail']['label'] == 'SYMPTOM':
            symptoms.append(f"{rel['head']['text']} {rel['tail']['text']}")
    allergens = [ent['text'] for ent in entities if ent['label'] == 'ALLERGEN']
    return highlights, ', '.join(toponyms), ', '.join(medicines), ', '.join(symptoms), ', '.join(allergens)

interface = gr.Interface(
    fn=analyze_text,
    inputs=gr.Textbox(lines=5, label="Текст"),
    outputs=[
        gr.HighlightedText(label="Сущности"),
        gr.Textbox(label="Топонимы"),
        gr.Textbox(label="Медицинские препараты"),
        gr.Textbox(label="Симптомы"),
        gr.Textbox(label="Аллергены"),
    ],
    title="PollenNER — извлечение сущностей и отношений",
    description="""Извлечение топонимов, лекарств, аллергенов, частей тела и симптомов из сообщений пользователей Пыльца Club.""",
    examples=[
        ["В Московской области у меня началась аллергия на пыльцу березы, потекли глаза, нос, принимаю Зиртек и Назонекс."],
        ["У ребенка в Новокузнецке чешутся глаза, уши и течет нос, врач прописал Кромогексал, Назонекс в нос."],
        ["В Санкт-Петербурге началось цветение ольхи, сильная реакция, принимаю Эриус, но глаза все равно слезятся."],
        ["В Калининграде вам будет намного лучше чем в Питере , местами есть пылящие березы ,но это уже как капля в море ."],
        ["Сергиев посад - слава Богу дожди. Дочь глаза чешет, горло чешется как бы тоже говорит (а так с марта на монтелукасте и зиртеке. Назонекс убрала, т.к кровь из носа часто идет."]
    ],
    cache_examples=True,
    flagging_mode="never"
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860, share=False, debug=True)