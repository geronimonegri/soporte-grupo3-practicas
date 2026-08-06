import os
import uuid

from flask import Flask, render_template, request, send_from_directory
from gtts import gTTS

app = Flask(__name__)

AUDIO_DIR = os.path.join(app.static_folder, "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

# Idiomas soportados por gTTS (código: nombre visible)
IDIOMAS = {
    "es": "Español",
    "en": "Inglés",
    "pt": "Portugués",
    "fr": "Francés",
    "it": "Italiano",
    "de": "Alemán",
}


@app.route("/", methods=["GET", "POST"])
def index():
    audio_file = None
    texto = ""
    idioma = "es"

    if request.method == "POST":
        texto = request.form.get("texto", "").strip()
        idioma = request.form.get("idioma", "es")

        if texto:
            nombre_archivo = f"{uuid.uuid4().hex}.mp3"
            ruta_completa = os.path.join(AUDIO_DIR, nombre_archivo)

            tts = gTTS(text=texto, lang=idioma)
            tts.save(ruta_completa)

            audio_file = nombre_archivo

    return render_template(
        "index.html",
        idiomas=IDIOMAS,
        audio_file=audio_file,
        texto=texto,
        idioma_seleccionado=idioma,
    )


@app.route("/audio/<nombre_archivo>")
def audio(nombre_archivo):
    return send_from_directory(AUDIO_DIR, nombre_archivo)


if __name__ == "__main__":
    app.run(debug=True)