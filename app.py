from flask import Flask, request, jsonify, render_template
import os
from dotenv import load_dotenv
import google.generativeai as genai
import nltk
import anthropic
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize
from google.cloud import translate_v2 as translate
app = Flask(__name__)

load_dotenv()
api_key = os.getenv('ANTHROPIC_API_KEY')
client = anthropic.Anthropic(
    api_key=api_key
)

app = Flask(__name__)

# Configure the API key for Google's Generative AI service
geminiapi_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=geminiapi_key)

# Model and generation config setup
generation_config = {
    "temperature": 0.5,
    "top_p": 0.95,
    "top_k": 100,
    "max_output_tokens": 8000,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
]

# Initialize the model with the specified configuration
model = genai.GenerativeModel(model_name="gemini-1.0-pro-001",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

def get_language_code(language_name):
    language_codes = {
        "English": "en",
        "Chinese (Simplified)": "zh-CN",
        "Chinese (Traditional)": "zh-TW",
        "Korean": "ko",
        "Japanese": "ja",
        "Spanish": "es",
        "French": "fr",
        "Russian": "ru",
        "Portuguese": "pt",
        "German": "de",
        "Arabic": "ar",
    }
    return language_codes.get(language_name, None)

def google_translate_text(sourcetext, target_language):
    translate_client = translate.Client()
    result = translate_client.translate(sourcetext, target_language=target_language)
    return result['translatedText']

def translate_text(sourcetext, inputlanguage, outputlanguage):
    
    try:
            translated_text = ""
            prompt = [
                f"Provide a accurate and quality translation of the following text from {inputlanguage} to {outputlanguage}. The translation should preserve cultural nuances, idiomatic expressions, and genre-specific terminology.",
                "In addition to translating the text itself, please ensure that the formatting and layout of the translated text mirrors the original text as closely as possible. This includes maintaining consistent naming conventions, terminology, and any specific formatting requirements particular to the genre or style of the text.",
                "There should be NO errors in the translaiton. The text should be more accurate than Google Translate.",
                "Input text: ",
                sourcetext, 
            ]
            response = model.generate_content(prompt)
            gemini_text = response.text
    except Exception as e:
            print(f"An error occurred while translating: {e}")
        
    return gemini_text

def pick_text_translator(sourcetext, inputlanguage, outputlanguage):
    
    output_language_code = get_language_code(outputlanguage)
    google_translated_text = google_translate_text(sourcetext, output_language_code)

    try:
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=4000,
            temperature=0.5,
            system=f"You are a professional translator tasked with combining two translations of a source text to create the most accurate and faithful translation possible. The source text is written in {inputlanguage} and needs to be translated into {outputlanguage}. You have been provided with two translations of the source text, Translation1 and Translation2. Your task is to compare and contrast these translations, and use your knowledge of both languages to produce a final translation that is as accurate and faithful to the source text as possible. Please be aware that accuracy is crucial, as any errors in the translation could have serious consequences. You should aim to preserve the tone, style, and meaning of the source text, while also ensuring that the translation is clear and natural in the target language.",
            messages=[
                {
                    "role": "user",
                    "content": f"Source text: {sourcetext}\nTranslation1: {google_translated_text}\nTranslation2: {translate_text(sourcetext, inputlanguage, outputlanguage)}"
                }
            ]
        )

        translated_text = response.content
        return translated_text
    except Exception as e:
        return print(f"An error occurred while translating: {e}")



@app.route('/', methods=['GET', 'POST'])
def index():
    translated_text = ""
    sourcetext = ""  # Initialize the variable to store source text
    inputlanguage = ""
    outputlanguage = ""
    if request.method == 'POST':
        sourcetext = request.form.get('sourceText')  # Capture the source text from the form
        inputlanguage = request.form.get('inputlanguage', 'Korean')  # Default to English if not specified
        outputlanguage = request.form.get('outputlanguage', 'English')  # Default to Spanish if not specified

        translated_text = translate_text(sourcetext, inputlanguage, outputlanguage)
        # Convert paragraph breaks to HTML line breaks
        translated_text = translated_text.replace('\n\n', '<br><br>')

    # Include sourcetext in the render_template context
    return render_template('index.html', sourcetext=sourcetext, translated_text=translated_text)

@app.route('/translate', methods=['POST'])
def translate_api():
    data = request.get_json()
    sourcetext = data.get('sourceText')
    inputlanguage = data.get('inputlanguage', 'Korean')  # Default to English if not specified
    outputlanguage = data.get('outputlanguage', 'English')  # Default to Spanish if not specified

    translated_text = translate_text(sourcetext, inputlanguage, outputlanguage)
    return jsonify({
        'sourceText': sourcetext,
        'translatedText': translated_text
    })

if __name__ == '__main__':
    app.run(debug=True)
