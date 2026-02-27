from flask import Flask, render_template, request
from transformers import pipeline
import requests

app = Flask(__name__)

classifier = pipeline(
    "text-classification",
    model="mrm8488/bert-tiny-finetuned-fake-news-detection"
)

def classify_news(text):
    result = classifier(text)[0]
    label = result['label']
    confidence = round(result['score'] * 100, 2)

    if label == "LABEL_0":
        final_label = "FAKE"
    else:
        final_label = "REAL"

    return final_label, confidence


def extract_claim(text):
    sentences = text.split(".")
    claim = sentences[0].strip()
    return claim


def verify_claim(claim):
    url = "https://en.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "list": "search",
        "srsearch": claim,
        "format": "json"
    }

    try:
        response = requests.get(url, params=params, timeout=5).json()
        results = response.get("query", {}).get("search", [])

        if len(results) > 0:
            return "Information found (Needs manual verification)"
        else:
            return "No reliable source found (Possibly Fake)"
    except:
        return "Verification service unavailable"

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    claim = None
    verification = None

    if request.method == "POST":
        news_text = request.form.get("news")

        if news_text:
            result, confidence = classify_news(news_text)
            claim = extract_claim(news_text)
            verification = verify_claim(claim)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        claim=claim,
        verification=verification
    )

if __name__ == "__main__":
    app.run(debug=True)