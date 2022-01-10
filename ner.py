import requests

entities_url = "https://eks-data-prod.onclusive.com/predictions/ner_neuron_b6"

text = "Tesla stock is on another tear for a few reasons, chief among is that CEO Elon Musk is essentially done selling his shares. But there’s another technical factor that appears to be driving shares higher into year-end: options trading."

r = requests.post(entities_url,
                  json={
                     'content': text,
                  },
                  timeout=120)

print(r.json())

#curl -H "Content-Type: application/json" -XPOST "https://eks-data-prod.onclusive.com/predictions/ner_neuron_b6" -d '{"content": "Tesla stock is on another tear for a few reasons, chief among is that CEO Elon Musk is essentially done selling his shares. dummy. But there’s another technical factor that appears to be driving shares higher into year-end: options trading. CEO Elon Musk is essentially done selling his shares."}'