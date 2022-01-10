from REL.mention_detection_base import MentionDetectionBase

def find_candidates(mention):
    base_url = '/home/ubuntu/REL/data'
    wiki_version = 'wiki_2019'

    mention_detector = MentionDetectionBase(base_url, wiki_version)
    cands = mention_detector.wiki_db.wiki(mention, "wiki")
    cands_cleaned = [c[0].replace("_", " ") for c in cands]
    return cands_cleaned

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