torchserve --stop
torch-model-archiver --force --model-name blink_entity_linking --version 1.0 --serialized-file ./blink_entity_linking.pt --handler "./handler_bert.py" --extra-files "./config.json" --export-path model_store
aws s3 cp model_store/blink_entity_linking.mar s3://airpr-dataprovider/models/neuron-archive/entity_linking/blink_entity_linking.mar