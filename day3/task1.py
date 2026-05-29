# EY Workshop | Participant: Jagat

from transformers import pipeline

# TODO 1: Initialise a sentiment-analysis pipeline
# Hint: classifier = pipeline("sentiment-analysis")
classifier = pipeline("sentiment-analysis")

# Sample sentences relevant to consulting work
sentences = [
    "The client was very satisfied with the delivery.",
    "The project is significantly over budget and behind schedule.",
    "The new regulatory framework presents both risks and opportunities.",
    # TODO 2: Add ONE sentence of your own make it work-relevant
    "Our team successfully delivered the digital transformation roadmap ahead of schedule."
]

# TODO 3: Run the classifier on all sentences and print results
# Expected output format:
# "The client was satisfied..." POSITIVE (0.9987)

def getSentiment(classifier, sentence):
    return classifier(sentence)

for sentence in sentences:
    result = getSentiment(classifier, sentence)
    label = result[0]['label']
    score = result[0]['score']
    print(f'"{sentence[:50]}..." {label} ({score:.4f})')
