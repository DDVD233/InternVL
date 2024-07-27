import json


def truncate_transcriptions(transcription_path: str, output_path: str) -> None:
    data = []  # jsonl
    with open(transcription_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    for i in range(len(data)):
        # Truncate everything after `The speaker said, '`, before `' What is the emotion of the speaker in this video?`
        # until it's under 300 words
        conversation_input = data[i]['conversations'][0]['value']
        if len(conversation_input) < 300 or 'The speaker said, \'' not in conversation_input:
            continue
        prefix, conversation_input = conversation_input.split('The speaker said, \'')
        conversation_input, suffix = conversation_input.split('\' What is the emotion of the speaker in this video?')
        if len(conversation_input) > 300:
            conversation_input = '...' + conversation_input[-300:]
        data[i]['conversations'][0]['value'] = (prefix + 'The speaker said, \''
                                                + conversation_input
                                                + '\' What is the emotion of the speaker in this video?'
                                                + suffix)
    with open(output_path, 'w') as f:
        for line in data:
            f.write(json.dumps(line) + '\n')


if __name__ == '__main__':
    truncate_transcriptions('val_annotations_phq.jsonl',
                            'val_annotations_phq_truncated.jsonl')