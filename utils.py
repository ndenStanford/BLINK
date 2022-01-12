from REL.mention_detection_base import MentionDetectionBase

def find_candidates(mention):
    base_url = '/home/ubuntu/REL/data'
    wiki_version = 'wiki_2019'

    mention_detector = MentionDetectionBase(base_url, wiki_version)
    cands = mention_detector.wiki_db.wiki(mention, "wiki")
    cands_cleaned = [c[0].replace("_", " ") for c in cands if c is not None]
    return cands_cleaned

def generate_sample(entities, sentences):
    entities = update_entities(entities, sentences)

    data_to_link = []
    _id = 0
    for entity in entities['entities']:
        mention = entity['entity_text']
        sentence_indexes = entity['sentence_indexes']
        entities_location_indexes = entity['sentence_indexes']
        for sentence_index, entities_location_index in zip(sentence_indexes, entities_location_indexes):
            context_left = sentences[sentence_index][:entities_location_index]
            context_right = sentences[sentence_index][entities_location_index+len(mention):]

            data_to_link += [{
                "id": _id,
                "label": "unknown",
                "label_id": -1,
                "context_left": context_left,
                "mention": mention,
                "context_right": context_right,
            }]
            _id += 1

    return data_to_link

def _sentence_indexes(match, texts):
        sentence_indexes = []
        entities_location_indexes = []
        ltok = len(match)
        for i, text in enumerate(texts):
            for j in range(len(text)):
                if match == text[j:(j+ltok)]:
                    sentence_indexes.append(i)
                    entities_location_indexes.append(j)
                    break
        return(sentence_indexes, entities_location_indexes)

def update_entities(entities, sentences):

    for i in range(len(entities['entities'])):
        match = entities['entities'][i]['entity_text']
        (sentence_indexes, entities_location_indexes) = _sentence_indexes(match, sentences)
        entities['entities'][i]['sentence_indexes'] = sentence_indexes
        entities['entities'][i]['entities_location_indexes'] = entities_location_indexes

    return entities