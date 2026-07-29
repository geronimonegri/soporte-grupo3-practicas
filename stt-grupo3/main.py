"""Traductor de voz express (STT + traducción) con Whisper + shortcut global.

Basado en el dictado por voz de ejemplo: F5 empieza a grabar; F5 de nuevo para,
transcribe con Whisper y TRADUCE el texto (español -> inglés) antes de pegarlo
donde tengas el foco (vía portapapeles + Cmd/Ctrl+V).

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
import sys
import threading

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
from deep_translator import GoogleTranslator

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

# El modelo por defecto depende de la plataforma: en Mac, el repo MLX local;
# en Windows/Linux, un tamaño de faster-whisper que descarga solo.
_DEFAULT_MODEL = (
    "mlx-community/whisper-large-v3-turbo" if IS_MAC else "large-v3"
)
MODEL = os.environ.get("WHISPER_MODEL", _DEFAULT_MODEL)

# Idioma de origen (lo que hablás) e idioma de destino (a lo que se traduce).
SRC_LANG = os.environ.get("WHISPER_LANG") or "es"
TARGET_LANG = os.environ.get("TARGET_LANG") or "en"

_kbd = Controller()
_translator = GoogleTranslator(source=SRC_LANG, target=TARGET_LANG)

# --- Motor de transcripción (se elige una vez, según la plataforma) -----------

if IS_MAC:
    import mlx_whisper

    def transcribe(audio):
        result = mlx_whisper.transcribe(
            audio, path_or_hf_repo=MODEL, language=SRC_LANG
        )
        return result["text"]
else:
    from faster_whisper import WhisperModel

    # compute_type="int8" corre bien en CPU; en GPU NVIDIA se puede usar "float16".
    _model = WhisperModel(MODEL, compute_type="int8")

    def transcribe(audio):
        segments, _ = _model.transcribe(audio, language=SRC_LANG)
        return "".join(seg.text for seg in segments)


def translate(text):
    """Traduce el texto de SRC_LANG a TARGET_LANG. Si falla, devuelve el original."""
    try:
        return _translator.translate(text)
    except Exception as e:
        print(f"(no se pudo traducir: {e})")
        return text


print(f"Plataforma: {'macOS (MLX)' if IS_MAC else sys.platform + ' (faster-whisper)'}")
print(f"Modelo: {MODEL}")
print(f"Traduciendo de '{SRC_LANG}' a '{TARGET_LANG}'.")
print(f"Listo. Pulsá {HOTKEY} para grabar/parar. Ctrl+C para salir.")

_frames = queue.Queue()
recording = False
# Serializa el toggle: impide arrancar una grabación nueva mientras el hilo
# anterior todavía está transcribiendo (los dos tocan _frames y `recording`).
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
    print(f"→ Original: {text!r}")
    if not text:
        return

    translated = translate(text).strip()
    print(f"→ Traducido: {translated!r}")

    if paste(translated):
        print("✔ Traducción pegada en la app activa.")
    else:
        print("✘ No se pudo pegar (¿falta permiso de Accesibilidad?). "
              "La traducción quedó en el portapapeles para pegar a mano.")


def _toggle_locked():
    # El lock asegura que un toggle termine (incluida la transcripción) antes
    # de que otro F5 arranque una grabación nueva. Evita que dos hilos se pisen
    # sobre _frames y `recording`.
    if not _lock.acquire(blocking=False):
        print("… ocupado, esperá a que termine.")
        return
    try:
        toggle()
    finally:
        _lock.release()


def on_press(key):
    if key == HOTKEY:
        # transcribir en otro hilo para no bloquear el listener de teclado
        threading.Thread(target=_toggle_locked, daemon=True).start()


# stream de micrófono siempre abierto; el flag `recording` decide si guardamos
with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=_callback):
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
