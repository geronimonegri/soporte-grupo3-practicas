import base64
import io
from flask import Flask, render_template, request
from gtts import gTTS

app = Flask(__name__)

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
    texto = ""
    idioma = "es"
    audio_base64 = None
    error_msg = None

    if request.method == "POST":
        texto = request.form.get("texto", "").strip()
        idioma = request.form.get("idioma", "es")
        if texto:
            try:
              
                fp = io.BytesIO()
                tts = gTTS(text=texto, lang=idioma)
                tts.write_to_fp(fp)
                fp.seek(0)
                audio_data = base64.b64encode(fp.read()).decode("utf-8")
                audio_base64 = f"data:audio/mp3;base64,{audio_data}"

            except Exception as e:
                error_msg = f"Error al generar el audio: {str(e)}"
    return render_template(
        "index.html",
        idiomas=IDIOMAS,
        texto=texto,
        idioma_seleccionado=idioma,
        audio_base64=audio_base64,
        error_msg=error_msg,
    )

if __name__ == "__main__":
    app.run(debug=True)