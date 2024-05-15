import json

with open('finance_sina.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

total_tokens = 0
entity_counts = {}

for item in data:
    text = item['text']
    labels = item['labels']
    total_tokens += len(labels)

    for label in labels:
        if label != 'O':
            entity_counts[label] = entity_counts.get(label, 0) + 1

entity_percentages = {label: count / total_tokens for label, count in entity_counts.items()}

print("All Tokens Numbers:", total_tokens)
print("rate of ner-label")
for label, percentage in entity_percentages.items():
    print(f"{label}: {percentage:.2%}")
