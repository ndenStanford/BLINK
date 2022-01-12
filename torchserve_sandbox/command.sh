torch-model-archiver --model-name entity_linking --version 1.0 --serialized-file ./dummy.pt --handler "./handler_bert.py" --extra-files "./config.json" --export-path model_store

torch-model-archiver --model-name entity_linking --version 1.0 --handler "./handler_bert.py" --extra-files "./config.json" --export-path model_store