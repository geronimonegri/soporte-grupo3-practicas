"""Asistente de voz simple: dictado + comandos (buscar en Google / escuchar en Spotify).

Basado en el dictado por voz de ejemplo: F5 empieza a grabar; F5 de nuevo para y
transcribe con Whisper. Según lo que dijiste, el programa hace una de estas 3 cosas:

  - Si empezás la frase con "buscar" / "busca" / "buscame" -> abre Google con
    una búsqueda de lo que dijiste después.
  - Si empezás la frase con "escuchar" / "poner" -> abre Spotify (búsqueda) con
    la canción que dijiste después. Si tenés la app instalada, se abre ahí
    directo; si no, se abre en el navegador. Queda a un clic de reproducir.
  - Si no dijiste ningún comando -> pega el texto tal cual donde tengas el
    cursor (dictado normal), como el TP de ejemplo original.

Motor de transcripción según la plataforma:
  - macOS (Apple Silicon): mlx-whisper sobre la GPU del Mac, usando los modelos
    MLX locales de ~/AI. No descarga nada.
  - Windows / Linux: faster-whisper (CPU o GPU NVIDIA). Descarga el modelo la
    primera vez y lo cachea.

Instalar:  pip install -r requirements.txt
macOS: dar permisos de Accesibilidad y Micrófono a la terminal en
       Ajustes > Privacidad y seguridad.
"""

import os
import queue
import re
import sys
import threading
import webbrowser
from urllib.parse import quote

# Detectar plataforma: en Mac usamos MLX; en el resto, faster-whisper.
IS_MAC = sys.platform == "darwin"

# Apuntar el cache de HuggingFace a ~/AI para que mlx-whisper encuentre el
# modelo local por nombre sin descargar. Debe ir ANTES de importar mlx_whisper.
if IS_MAC:
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/AI"))

import numpy as np
import sounddevice as sd
import pyperclip
from pynput import keyboard
from pynput.keyboard import Controller, Key

# .env leído a mano (pocos valores) para no sumar python-dotenv.
_env = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env):
    for line in open(_env):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

SAMPLE_RATE = 16000  # whisper trabaja a 16kHz
HOTKEY = keyboard.Key.f5

_DEFAULT_MODEL = (
    "mlx-community/whisper-large-v3-turbo" if IS_MAC else "large-v3"
)
MODEL = os.environ.get("WHISPER_MODEL", _DEFAULT_MODEL)
LANG = os.environ.get("WHISPER_LANG") or "es"

_kbd = Controller()

# --- Comandos reconocidos al principio de la frase -----------------------
# "buscar/busca/buscame algo"      -> Google
# "escuchar/poner algo"            -> Spotify
_BUSCAR_RE = re.compile(r"^\s*(buscar|busca|buscame)\s+(.+)", re.IGNORECASE)
_ESCUCHAR_RE = re.compile(r"^\s*(escuchar|poner)\s+(.+)", re.IGNORECASE)


def buscar_google(query):
    url = f"https://www.google.com/search?q={quote(query)}"
    webbrowser.open(url)


def buscar_spotify(cancion):
    # spotify: abre la app de escritorio directo en la búsqueda, si está instalada.
    # Si no hay app asociada al esquema "spotify:", el sistema puede ignorarlo;
    # por eso además abrimos el buscador web como respaldo.
    webbrowser.open(f"spotify:search:{quote(cancion)}")
    webbrowser.open(f"https://open.spotify.com/search/{quote(cancion)}")


def procesar_texto(text):
    """Decide qué hacer según el texto transcripto. Devuelve un mensaje para consola."""
    m = _BUSCAR_RE.match(text)
    if m:
        query = m.group(2).strip()
        buscar_google(query)
        return f"🔎 Buscando en Google: {query!r}"

    m = _ESCUCHAR_RE.match(text)
    if m:
        cancion = m.group(2).strip()
        buscar_spotify(cancion)
        return f"🎵 Abriendo en Spotify: {cancion!r}"

    # Sin comando reconocido: dictado normal, pega el texto donde tengas el cursor.
    if paste(text):
        return "✔ Texto pegado en la app activa."
    return ("✘ No se pudo pegar (¿falta permiso de Accesibilidad?). "
            "El texto quedó en el portapapeles para pegar a mano.")


# --- Motor de transcripción (se elige una vez, según la plataforma) -----------

if IS_MAC:
    import mlx_whisper

    def transcribe(audio):
        result = mlx_whisper.transcribe(
            audio, path_or_hf_repo=MODEL, language=LANG
        )
        return result["text"]
else:
    from faster_whisper import WhisperModel

    _model = WhisperModel(MODEL, compute_type="int8")

    def transcribe(audio):
        segments, _ = _model.transcribe(audio, language=LANG)
        return "".join(seg.text for seg in segments)


print(f"Plataforma: {'macOS (MLX)' if IS_MAC else sys.platform + ' (faster-whisper)'}")
print(f"Modelo: {MODEL}")
print("Comandos: 'buscar <algo>' -> Google | 'escuchar <cancion>' -> Spotify")
print(f"Listo. Pulsá {HOTKEY} para grabar/parar. Ctrl+C para salir.")

_frames = queue.Queue()
recording = False
_lock = threading.Lock()


def _callback(indata, frames, time, status):
    if recording:
        _frames.put(indata.copy())


def paste(text):
    """Deja el texto en el portapapeles y simula Cmd/Ctrl+V. Devuelve True si pegó."""
    try:
        pyperclip.copy(text)
        modifier = Key.cmd if IS_MAC else Key.ctrl
        with _kbd.pressed(modifier):
            _kbd.press("v")
            _kbd.release("v")
        return True
    except Exception:
        return False


def toggle():
    global recording
    if not recording:
        while not _frames.empty():  # limpiar audio viejo
            _frames.get()
        recording = True
        print("● Grabando...")
        return

    recording = False
    print("■ Transcribiendo...")
    chunks = []
    while not _frames.empty():
        chunks.append(_frames.get())
    if not chunks:
        print("(sin audio)")
        return

    audio = np.concatenate(chunks).flatten().astype(np.float32)
    text = transcribe(audio).strip()
    print(f"→ {text!r}")
    if not text:
        return

    resultado = procesar_texto(text)
    print(resultado)


def _toggle_locked():
    if not _lock.acquire(blocking=False):
        print("… ocupado, esperá a que termine.")
        return
    try:
        toggle()
    finally:
        _lock.release()


def on_press(key):
    if key == HOTKEY:
        threading.Thread(target=_toggle_locked, daemon=True).start()


with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=_callback):
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
