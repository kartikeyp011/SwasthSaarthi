from fastapi import FastAPI, HTTPException
import requests
import os

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_API_URL = "https://api-inference.huggingface.co/models/facebook/nllb-200-distilled-600M"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# Language Mapping
LANGUAGE_CODES = {  # Use the dictionary from above
    "Assamese": "asm_Beng",
    "Bengali": "ben_Beng",
    "Bhojpuri": "bho_Deva",
    "Chhattisgarhi": "hne_Deva",
    "Gujarati": "guj_Gujr",
    "Hindi": "hin_Deva",
    "Kannada": "kan_Knda",
    "Kashmiri (Arabic)": "kas_Arab",
    "Kashmiri (Devanagari)": "kas_Deva",
    "Konkani": "kon_Deva",
    "Maithili": "mai_Deva",
    "Malayalam": "mal_Mlym",
    "Marathi": "mar_Deva",
    "Meitei (Bengali)": "mni_Beng",
    "Nepali": "npi_Deva",
    "Oriya": "ory_Orya",
    "Punjabi": "pan_Guru",
    "Sanskrit": "san_Deva",
    "Santali": "sat_Olck",
    "Sindhi": "snd_Arab",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Urdu": "urd_Arab"
}

@app.post("/translate")
async def translate_text(text: str, target_language: str):
    """ Translates text into the user’s desired language """
    if target_language not in LANGUAGE_CODES:
        raise HTTPException(status_code=400, detail="Invalid target language")

    target_lang_code = LANGUAGE_CODES[target_language]
    
    payload = {
        "inputs": text,
        "parameters": {"src_lang": "eng_Latn", "tgt_lang": target_lang_code}
    }

    response = requests.post(HF_API_URL, headers=HEADERS, json=payload)

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Translation API error")

    translated_text = response.json()[0]["translation_text"]
    return {"translated_text": translated_text}