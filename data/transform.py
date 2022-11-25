import json

for split in ['train', 'dev', 'test']:
    with open(f'{split}-v2.0.json', 'r') as fr:
        data = []
        source = json.load(fr)
        for article in source['data']:
            for para in article['paragraphs']:
                for qa in para['qas']:
                    answers = {'text': [], 'answer_start': []}
                    for answer in qa['answers']:
                        answers['text'].append(answer['text'])
                        answers['answer_start'].append(answer['answer_start'])
                    data.append({
                        'id': qa['id'],
                        'context': para['context'],
                        'question': qa['question'],
                        'answers': answers
                    })
    with open(f'{split}-v2.0.hf.json', 'w') as fw:
        json.dump(data, fw)
        