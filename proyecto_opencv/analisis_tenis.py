"""
Análisis de tenis profesional — MOG2 + máscara de polígono + perspectiva
=========================================================================
Detecta jugadores con MOG2 dentro del trapecio exacto de la cancha.
Mapea posiciones a cancha.png con transformación de perspectiva real.

Uso:
    python analisis_tenis.py video.mp4

Salida:
    - trayectorias.png
    - mapa_calor.png

Requisitos:
    pip install opencv-python numpy matplotlib
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from pathlib import Path
import sys
import os
import urllib.request

# ============================================================
# CONFIGURACIÓN
# ============================================================

FRAME_SKIP = 8

# Vista cenital
HSV_CANCHA_BAJO = np.array([85,  40,  60])
HSV_CANCHA_ALTO = np.array([160, 255, 255])
UMBRAL_CENITAL  = 0.25

# Zona de tracking — máscara de exclusión (rectángulo rojo)
ZONA_TRACKING = np.array([
    [606,  160],
    [1380, 180],
    [1782, 980],
    [262,  960],
], dtype=np.int32)

# Cancha exacta — solo para mapeo de perspectiva a cancha.png (trapecio verde)
CANCHA_PUNTOS = np.array([
    [684,  250],   # Superior izquierda
    [1240, 270],   # Superior derecha
    [1506, 820],   # Inferior derecha
    [408,  808],   # Inferior izquierda
], dtype=np.int32)

# Red — separa Jugador A (cercano) de Jugador B (lejano)
RED_IZQ = (600,  416)
RED_DER = (1326, 426)

# Zona del árbitro a excluir (silla, borde izquierdo)
ARBITRO_X_MAX = 330
ARBITRO_Y_MAX = 400

# YOLO — detección de jugadores
YOLO_MODEL_PATH   = "yolov8n.onnx"
YOLO_MODEL_URL    = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx"
YOLO_CONF_THRESH  = 0.4
YOLO_ASPECT_MIN   = 0.8    # h/w mínimo: persona parada es más alta que ancha

# Filtro temporal de jugadores
JUGADOR_MAX_SALTO = 150    # px en frame original
JUGADOR_MAX_PERD  = 5

# MOG2
MOG2_HISTORY       = 500
MOG2_VAR_THRESHOLD = 40

# Pelota — HSV amarillo-verde
PELOTA_HSV_BAJO  = np.array([29, 86,  6])
PELOTA_HSV_ALTO  = np.array([64, 255, 255])
PELOTA_AREA_MIN  = 10
PELOTA_AREA_MAX  = 200
PELOTA_MAX_SALTO = 100

# Golpes
ANGULO_GOLPE   = 60
COOLDOWN_GOLPE = 10

# Colas de trayectoria en el video
COLA_JUGADORES = 20
COLA_PELOTA    = 15

# Imagen final
TRAY_MAX_SALTO = 120

# Colores BGR
COLOR_A      = (0,   0,  220)
COLOR_B      = (200, 80,   0)
COLOR_BBOX_A = (180, 100, 255)   # rosa
COLOR_BBOX_B = (255, 160,  50)   # celeste
COLOR_PELOTA = (0,  220, 255)    # amarillo
COLOR_GOLPE  = (0,  220, 255)
COLOR_CANCHA = (0,  200,   0)    # verde — borde del polígono
COLOR_TEXTO  = (255, 255, 255)


# ============================================================
# MÓDULO 0 — VISTA CENITAL
# ============================================================

def es_cenital(pequeño_hsv):
    """True si el frame (a mitad de resolución) tiene suficiente cancha visible."""
    mask = cv2.inRange(pequeño_hsv, HSV_CANCHA_BAJO, HSV_CANCHA_ALTO)
    return np.count_nonzero(mask) / mask.size > UMBRAL_CENITAL


# ============================================================
# MÓDULO 1 — DETECCIÓN DE JUGADORES CON YOLO
# ============================================================

def y_red_en_x(x):
    """Interpola la y de la red en la posición x del centroide."""
    t = (x - RED_IZQ[0]) / (RED_DER[0] - RED_IZQ[0])
    return RED_IZQ[1] + t * (RED_DER[1] - RED_IZQ[1])


def _cargar_yolo():
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"ERROR: No se encuentra {YOLO_MODEL_PATH}")
        print("Descargá el modelo manualmente desde:")
        print("https://github.com/ultralytics/assets/releases/latest/download/yolov8n.onnx")
        print(f"y guardalo en la carpeta del proyecto como '{YOLO_MODEL_PATH}'")
        sys.exit(1)
    net = cv2.dnn.readNetFromONNX(YOLO_MODEL_PATH)
    print(f"  YOLOv8n cargado: {YOLO_MODEL_PATH}")
    return net

def detectar_con_yolo(frame, net, conf_thresh=YOLO_CONF_THRESH):
    """
    Corre YOLOv8n sobre el frame y retorna lista de bounding boxes
    de personas (clase 0) con confianza > conf_thresh.
    Retorna lista de (x1, y1, x2, y2, confianza).
    """
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (640, 640),
                                  swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward()
    # YOLOv8 output: (1, 84, 8400) → outputs[0] es (84, 8400)
    out = outputs[0]

    boxes = []
    for i in range(out.shape[1]):
        scores   = out[4:, i]
        class_id = int(np.argmax(scores))
        conf     = float(scores[class_id])
        if class_id != 0 or conf < conf_thresh:
            continue
        cx, cy, bw, bh = out[0, i], out[1, i], out[2, i], out[3, i]
        x1 = int((cx - bw / 2) / 640 * w)
        y1 = int((cy - bh / 2) / 640 * h)
        x2 = int((cx + bw / 2) / 640 * w)
        y2 = int((cy + bh / 2) / 640 * h)
        boxes.append((x1, y1, x2, y2, conf))
    return boxes


def separar_jugadores_yolo(boxes):
    """
    Toma la lista de (x1,y1,x2,y2,conf) de YOLO y retorna
    (torso_A, zap_A, bbox_A, conf_A, torso_B, zap_B, bbox_B, conf_B).
    Filtra por centroide en ZONA_TRACKING, aspect ratio h/w >= YOLO_ASPECT_MIN
    y separación respecto a la línea de red con margen ±20 px.
    """
    cands_A, cands_B = [], []

    for (x1, y1, x2, y2, conf) in boxes:
        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0 or bh <= 0:
            continue

        if bh / bw < YOLO_ASPECT_MIN:
            continue

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if cv2.pointPolygonTest(ZONA_TRACKING, (float(cx), float(cy)), False) < 0:
            continue

        if cx < ARBITRO_X_MAX and cy < ARBITRO_Y_MAX:
            continue

        torso = (cx, cy)
        zap   = (cx, y2)
        bbox  = (x1, y1, x2, y2)
        area  = bw * bh

        y_red = y_red_en_x(cx)
        if cy > y_red + 20:
            cands_A.append((area, torso, zap, bbox, conf))
        elif cy < y_red - 20:
            cands_B.append((area, torso, zap, bbox, conf))

    def mejor(cands):
        if not cands:
            return None, None, None, None
        _, torso, zap, bbox, conf = max(cands, key=lambda c: c[0])
        return torso, zap, bbox, conf

    torso_A, zap_A, bbox_A, conf_A = mejor(cands_A)
    torso_B, zap_B, bbox_B, conf_B = mejor(cands_B)
    return torso_A, zap_A, bbox_A, conf_A, torso_B, zap_B, bbox_B, conf_B


def filtrar_temporal(pos_nueva, ultima_pos, frames_sin):
    """
    Descarta detecciones con salto imposible (> JUGADOR_MAX_SALTO).
    Resetea la posición conocida tras JUGADOR_MAX_PERD frames sin detección válida.
    """
    if pos_nueva is not None:
        if ultima_pos is not None:
            d = np.hypot(pos_nueva[0] - ultima_pos[0],
                         pos_nueva[1] - ultima_pos[1])
            if d > JUGADOR_MAX_SALTO:
                frames_sin += 1
                if frames_sin > JUGADOR_MAX_PERD:
                    ultima_pos = None
                return None, ultima_pos, frames_sin
        return pos_nueva, pos_nueva, 0
    else:
        frames_sin += 1
        if frames_sin > JUGADOR_MAX_PERD:
            ultima_pos = None
        return None, ultima_pos, frames_sin


# ============================================================
# MÓDULO 2 — DETECCIÓN DE PELOTA
# ============================================================

def detectar_pelota(frame_hsv, mascara_cancha, ultima_pelota=None, mask_mog=None):
    """
    Detecta la pelota por filtro HSV dentro del polígono de la cancha.
    Si mask_mog se provee (MOG2), lo aplica como filtro adicional de movimiento.
    Aplica consistencia temporal: no puede saltar más de PELOTA_MAX_SALTO px.
    """
    mask = cv2.inRange(frame_hsv, PELOTA_HSV_BAJO, PELOTA_HSV_ALTO)
    mask = cv2.bitwise_and(mask, mascara_cancha)
    if mask_mog is not None:
        mask = cv2.bitwise_and(mask, mask_mog)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    mejor = None
    mejor_score = 0

    for c in contornos:
        area = cv2.contourArea(c)
        if not (PELOTA_AREA_MIN <= area <= PELOTA_AREA_MAX):
            continue
        per = cv2.arcLength(c, True)
        if per == 0:
            continue
        circ = 4 * np.pi * area / (per ** 2)
        if circ < 0.4:
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        if ultima_pelota is not None:
            if np.hypot(cx - ultima_pelota[0], cy - ultima_pelota[1]) > PELOTA_MAX_SALTO:
                continue

        score = circ / (area + 1)
        if score > mejor_score:
            mejor_score = score
            mejor = (cx, cy)

    return mejor


def detectar_golpe(hist_pelota, ultimo_golpe_p, proc_actual):
    """True si la pelota cambió de dirección más de ANGULO_GOLPE grados."""
    if proc_actual - ultimo_golpe_p < COOLDOWN_GOLPE:
        return False
    puntos = [p for p in hist_pelota if p is not None]
    if len(puntos) < 3:
        return False
    v1 = np.array([puntos[-2][0] - puntos[-3][0],
                   puntos[-2][1] - puntos[-3][1]], dtype=float)
    v2 = np.array([puntos[-1][0] - puntos[-2][0],
                   puntos[-1][1] - puntos[-2][1]], dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 2 or n2 < 2:
        return False
    angulo = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)))
    return angulo > ANGULO_GOLPE


# ============================================================
# MÓDULO 3 — TRANSFORMACIÓN DE PERSPECTIVA
# ============================================================


def crear_transformacion(img_cancha_rgb, rect_cancha_img=None):
    H, W = img_cancha_rgb.shape[:2]

    if rect_cancha_img is not None:
        rx, ry, rw, rh = rect_cancha_img
        top    = ry
        bottom = ry + rh
        left   = rx
        right  = rx + rw
    else:
        top, bottom = int(H * 0.05), int(H * 0.95)
        left, right = int(W * 0.05), int(W * 0.95)

    # ORDEN: sup-izq, sup-der, inf-der, inf-izq
    pts_video = np.float32([
        [684,  238],   # sup-izq → Nadal, fondo lejano izquierda
        [1240, 248],   # sup-der → Nadal, fondo lejano derecha
        [1506, 764],   # inf-der → Djokovic, fondo cercano derecha
        [408,  752],   # inf-izq → Djokovic, fondo cercano izquierda
    ])

    pts_cancha = np.float32([
        [left,  top],      # Nadal fondo lejano izq → arriba-izq
        [right, top],      # Nadal fondo lejano der → arriba-der
        [right, bottom],   # Djokovic fondo cercano der → abajo-der
        [left,  bottom],   # Djokovic fondo cercano izq → abajo-izq
    ])

    print(f"  pts_cancha: top={top} bottom={bottom} left={left} right={right}")
    return cv2.getPerspectiveTransform(pts_video, pts_cancha)


def video_a_cancha(x, y, M):
    """Mapea un punto (x, y) del video a coordenadas de cancha.png."""
    pt  = np.float32([[[float(x), float(y)]]])
    res = cv2.perspectiveTransform(pt, M)
    return int(res[0][0][0]), int(res[0][0][1])


# ============================================================
# MÓDULO 5 — DIBUJO EN VIDEO
# ============================================================

def dibujar_cola(frame, cola, color, radio_max=4):
    pts = list(cola)
    n   = len(pts)
    for i, p in enumerate(pts):
        if p is None:
            continue
        alpha = (i + 1) / n
        c = tuple(int(ch * alpha) for ch in color)
        cv2.circle(frame, p, max(1, int(radio_max * alpha)), c, -1)


def dibujar_jugador(frame, torso, zap, color, nombre):
    if torso is not None and zap is not None:
        cv2.line(frame, torso, zap, color, 2, cv2.LINE_AA)
        cv2.circle(frame, torso, 15, (0, 0, 200),     -1)   # rojo — torso
        cv2.circle(frame, zap,    8, (255, 255, 255),  -1)   # blanco — zapatillas
        cv2.putText(frame, nombre, (torso[0] + 18, torso[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def dibujar_overlay(frame, frame_num, fps, golpes, torso_A, torso_B):
    h  = frame.shape[0]
    px, py = 8, h - 80

    ov = frame.copy()
    cv2.rectangle(ov, (px, py), (px + 400, h - 8), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)

    t = frame_num / fps

    def txt(texto, dy, color=COLOR_TEXTO):
        cv2.putText(frame, texto, (px + 8, py + dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

    txt(f"t = {t:.1f}s  |  Golpes: {golpes}", 22)
    txt(f"Jugador A: {'detectado' if torso_A else 'no detectado'}", 46, COLOR_BBOX_A)
    txt(f"Jugador B: {'detectado' if torso_B else 'no detectado'}", 68, COLOR_BBOX_B)


# ============================================================
# UTILIDAD — RECT DE CANCHA EN cancha.png
# ============================================================

def detectar_rect_cancha_imagen(img_cancha_rgb):
    """Bounding rect de los píxeles blancos (líneas) en cancha.png."""
    gris = cv2.cvtColor(img_cancha_rgb, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gris, 200, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    alto_img, ancho_img = img_cancha_rgb.shape[:2]
    if w < ancho_img * 0.2 or h < alto_img * 0.2:
        return None
    return x, y, w, h


# ============================================================
# MÓDULO 6 — IMÁGENES FINALES
# ============================================================

def conclusion_tactica(ys_cancha, alto_c, nombre):
    if not ys_cancha:
        return f"{nombre}: sin datos"
    n = len(ys_cancha)
    pct_fondo = sum(1 for y in ys_cancha if y > alto_c * 0.65) / n
    pct_red   = sum(1 for y in ys_cancha if y < alto_c * 0.35) / n
    if pct_fondo > 0.55:
        return f"{nombre} jugó principalmente desde el fondo"
    if pct_red > 0.35:
        return f"{nombre} subió frecuentemente a la red"
    return f"{nombre} dominó la zona media de la cancha"


def guardar_trayectorias(tray_A, tray_B, img_cancha_rgb,
                         frames_A, frames_B, ruta="trayectorias.png"):
    if len(tray_A) < 2 and len(tray_B) < 2:
        print("  No hay datos suficientes para trayectorias.png")
        return

    alto_c, ancho_c = img_cancha_rgb.shape[:2]
    fig, ax = plt.subplots(figsize=(6, 6 * alto_c / ancho_c))
    fig.patch.set_facecolor("#0d1b2a")
    ax.imshow(img_cancha_rgb, zorder=1)
    ax.set_xlim(0, ancho_c)
    ax.set_ylim(alto_c, 0)

    for tray, rgb, etiqueta in [
        (tray_A, (0.85, 0.1, 0.1),  f"Jugador A ({frames_A} frames)"),
        (tray_B, (0.1,  0.3, 0.95), f"Jugador B ({frames_B} frames)"),
    ]:
        if len(tray) < 2:
            continue
        n = len(tray)
        for i in range(1, n):
            dist = np.hypot(tray[i][0] - tray[i-1][0],
                            tray[i][1] - tray[i-1][1])
            if dist > TRAY_MAX_SALTO:
                continue
            t = i / n
            color_seg = (rgb[0] * t,
                         rgb[1] * t + 0.1 * (1 - t),
                         rgb[2] * t + 0.2 * (1 - t))
            ax.plot([tray[i-1][0], tray[i][0]],
                    [tray[i-1][1], tray[i][1]],
                    color=color_seg, linewidth=1.0 + t * 1.5,
                    alpha=0.75, solid_capstyle="round", zorder=3)
        ax.scatter(tray[0][0],  tray[0][1],  s=60, color="white",
                   edgecolors=rgb, linewidths=1.5, zorder=5,
                   label=f"{etiqueta} — inicio")
        ax.scatter(tray[-1][0], tray[-1][1], s=80, color=rgb,
                   edgecolors="white", linewidths=1.5, zorder=5,
                   label=f"{etiqueta} — fin")

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.12),
              ncol=2, fontsize=8, frameon=True,
              facecolor="#0d1b2a", edgecolor="white", labelcolor="white")
    ax.set_title("Trayectorias del partido",
                 color="white", fontsize=13, fontweight="bold", pad=10)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(ruta, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Guardado: {ruta}")


def guardar_mapa_calor(calor_A, calor_B, ys_A, ys_B,
                       img_cancha_rgb, alto_c,
                       rect_cancha_img=None, ruta="mapa_calor.png"):
    def blur_norm(acc):
        if acc.max() == 0:
            return None
        b = cv2.GaussianBlur(acc, (81, 81), 0)
        return b / b.max()

    def to_disp(arr):
        if arr is None:
            return None
        d = arr.astype(float)
        d[d < 0.02] = np.nan
        return d

    concl = (f"• {conclusion_tactica(ys_A, alto_c, 'Jugador A')}\n"
             f"• {conclusion_tactica(ys_B, alto_c, 'Jugador B')}")

    alto_ci, ancho_ci = img_cancha_rgb.shape[:2]
    fig, ax = plt.subplots(figsize=(6, 6 * alto_ci / ancho_ci))
    fig.patch.set_facecolor("#0d1b2a")
    ax.imshow(img_cancha_rgb, zorder=1)
    ax.set_xlim(0, ancho_ci)
    ax.set_ylim(alto_ci, 0)

    # Mostrar heatmap en toda la imagen, sin limitar al rect de la cancha
    def preparar(acc):
        h = blur_norm(acc)
        if h is None:
            return None
        return to_disp(h)

    for acc, cmap_name in [(calor_A, "Reds"), (calor_B, "Blues")]:
        heat = preparar(acc)
        if heat is None:
            continue
        cmap = plt.colormaps[cmap_name].copy()
        cmap.set_bad(alpha=0.0)
        ax.imshow(heat, cmap=cmap, alpha=0.65, vmin=0, vmax=1,
                  zorder=2, origin="upper",
                  extent=[0, ancho_ci, alto_ci, 0])

    ax.set_title("Mapa de calor de posición",
                 color="white", fontsize=13, fontweight="bold", pad=10)
    ax.axis("off")
    fig.text(0.5, 0.01, concl, ha="center", color="white",
             fontsize=9, linespacing=1.6,
             bbox=dict(boxstyle="round,pad=0.4",
                       facecolor="#0d1b2a", edgecolor="#555555"))
    plt.tight_layout()
    plt.savefig(ruta, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Guardado: {ruta}")


def guardar_resumen_combinado(tray_A, tray_B, calor_A, calor_B,
                               img_cancha_rgb, frames_A, frames_B,
                               alto_c, rect_cancha_img=None,
                               ruta="resumen.png"):
    alto_ci, ancho_ci = img_cancha_rgb.shape[:2]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6 * alto_ci / ancho_ci))
    fig.patch.set_facecolor("#0d1b2a")

    # ---- Panel izquierdo: Trayectorias ----
    ax1.imshow(img_cancha_rgb, zorder=1)
    ax1.set_xlim(0, ancho_ci)
    ax1.set_ylim(alto_ci, 0)
    ax1.set_title("Trayectorias", color="white", fontsize=11, fontweight="bold", pad=8)
    ax1.axis("off")

    for tray, rgb, etiqueta in [
        (tray_A, (0.85, 0.1, 0.1), f"Jugador A ({frames_A} frames)"),
        (tray_B, (0.1,  0.3, 0.95), f"Jugador B ({frames_B} frames)"),
    ]:
        if len(tray) < 2:
            continue
        n = len(tray)
        for i in range(1, n):
            dist = np.hypot(tray[i][0]-tray[i-1][0], tray[i][1]-tray[i-1][1])
            if dist > TRAY_MAX_SALTO:
                continue
            t = i / n
            color_seg = (rgb[0]*t, rgb[1]*t + 0.1*(1-t), rgb[2]*t + 0.2*(1-t))
            ax1.plot([tray[i-1][0], tray[i][0]], [tray[i-1][1], tray[i][1]],
                     color=color_seg, linewidth=1.0 + t*1.5,
                     alpha=0.75, solid_capstyle="round", zorder=3)
        ax1.scatter(tray[0][0],  tray[0][1],  s=60, color="white",
                    edgecolors=rgb, linewidths=1.5, zorder=5, label=f"{etiqueta} — inicio")
        ax1.scatter(tray[-1][0], tray[-1][1], s=80, color=rgb,
                    edgecolors="white", linewidths=1.5, zorder=5, label=f"{etiqueta} — fin")
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.08),
               ncol=2, fontsize=7, frameon=True,
               facecolor="#0d1b2a", edgecolor="white", labelcolor="white")

    # ---- Panel derecho: Mapa de calor ----
    ax2.imshow(img_cancha_rgb, zorder=1)
    ax2.set_xlim(0, ancho_ci)
    ax2.set_ylim(alto_ci, 0)
    ax2.set_title("Mapa de calor", color="white", fontsize=11, fontweight="bold", pad=8)
    ax2.axis("off")

    def blur_norm(acc):
        if acc.max() == 0:
            return None
        b = cv2.GaussianBlur(acc, (81, 81), 0)
        return b / b.max()

    def to_disp(arr):
        if arr is None:
            return None
        d = arr.astype(float)
        d[d < 0.02] = np.nan
        return d

    for acc, cmap_name in [(calor_A, "Reds"), (calor_B, "Blues")]:
        heat = to_disp(blur_norm(acc))
        if heat is None:
            continue
        cmap = plt.colormaps[cmap_name].copy()
        cmap.set_bad(alpha=0.0)
        ax2.imshow(heat, cmap=cmap, alpha=0.65, vmin=0, vmax=1,
                   zorder=2, origin="upper", extent=[0, ancho_ci, alto_ci, 0])

    # ---- Título general ----
    fig.suptitle("Análisis del partido — OpenCV + YOLOv8",
                 color="white", fontsize=13, fontweight="bold", y=1.01)

    plt.tight_layout()
    plt.savefig(ruta, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Guardado: {ruta}")


# ============================================================
# MÓDULO 7 — PREVIEW DE CANCHA EN TIEMPO REAL
# ============================================================

def render_cancha_preview(cancha_img, calor_A, calor_B, tray_A, tray_B):
    """Genera imagen BGR con heatmap + trayectorias para cv2.imshow."""
    preview = cancha_img.copy()

    for calor, color_map in [(calor_A, cv2.COLORMAP_HOT),
                             (calor_B, cv2.COLORMAP_WINTER)]:
        if calor.max() > 0:
            norm = cv2.normalize(calor, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            blur = cv2.GaussianBlur(norm, (51, 51), 0)
            heat = cv2.applyColorMap(blur, color_map)
            mask = blur > 15
            preview[mask] = cv2.addWeighted(preview, 0.4, heat, 0.6, 0)[mask]

    for tray, color in [(tray_A, (0, 0, 220)), (tray_B, (220, 80, 0))]:
        for i in range(1, len(tray)):
            if np.hypot(tray[i][0] - tray[i-1][0],
                        tray[i][1] - tray[i-1][1]) < 60:
                cv2.line(preview, tray[i-1], tray[i], color, 2)

    return preview


def dibujar_minimap(cancha_img, calor_A, calor_B, tray_A, tray_B):
    """Minimap landscape (420×280) con heatmap y trayectorias para overlay en el video."""
    mini_w, mini_h = 420, 280
    preview = render_cancha_preview(cancha_img, calor_A, calor_B, tray_A, tray_B)
    return cv2.resize(preview, (mini_w, mini_h))


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def procesar_video(ruta_entrada, modo_test=False):
    cap = cv2.VideoCapture(ruta_entrada)
    if not cap.isOpened():
        print(f"ERROR: No se puede abrir '{ruta_entrada}'")
        sys.exit(1)

    fps_orig = cap.get(cv2.CAP_PROP_FPS)
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret_t, frame_t = cap.read()
    if not ret_t:
        print("ERROR: No se puede leer el primer frame.")
        sys.exit(1)
    alto_v, ancho_v = frame_t.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # YOLO — cargar una vez antes del loop
    net = _cargar_yolo()

    # Máscara de tracking (siempre necesaria)
    mascara_cancha = np.zeros((alto_v, ancho_v), dtype=np.uint8)
    cv2.fillPoly(mascara_cancha, [ZONA_TRACKING], 255)

    # MOG2 — solo para pelota (objetos pequeños en movimiento)
    sustractor = cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY,
        varThreshold=MOG2_VAR_THRESHOLD,
        detectShadows=False
    )

    # Cancha — necesaria en ambos modos (perspectiva + preview)
    cancha_path = Path("cancha.png")
    if cancha_path.exists():
        img_cancha_bgr = cv2.imread(str(cancha_path))
        img_cancha_rgb = cv2.cvtColor(img_cancha_bgr, cv2.COLOR_BGR2RGB)
        print(f"  cancha.png: {img_cancha_rgb.shape[1]}×{img_cancha_rgb.shape[0]} px")
    else:
        print("  AVISO: cancha.png no encontrada — fondo genérico")
        img_cancha_rgb = np.full((900, 500, 3), 30, dtype=np.uint8)
        img_cancha_bgr = img_cancha_rgb.copy()
    alto_c, ancho_c = img_cancha_rgb.shape[:2]

    rect_cancha_img = detectar_rect_cancha_imagen(img_cancha_rgb)
    if rect_cancha_img:
        print(f"  Rect cancha en imagen: {rect_cancha_img}")
    M_perspectiva   = crear_transformacion(img_cancha_rgb, rect_cancha_img)

    calor_A    = np.zeros((alto_c, ancho_c), dtype=np.float32)
    calor_B    = np.zeros((alto_c, ancho_c), dtype=np.float32)
    tray_A_all = []
    tray_B_all = []

    # Inicialización solo para modo normal
    if not modo_test:
        cola_A      = deque(maxlen=COLA_JUGADORES)
        cola_B      = deque(maxlen=COLA_JUGADORES)
        cola_pelota = deque(maxlen=COLA_PELOTA)

        ultima_pelota  = None
        hist_pelota    = deque(maxlen=5)
        ultimo_golpe_p = -(COOLDOWN_GOLPE + 1)
        golpe_pos      = None

        golpes_count = 0

    ultima_A_p     = None;  frames_sin_A = 0
    ultima_B_p     = None;  frames_sin_B = 0
    frames_cenital = 0
    frames_A       = 0
    frames_B       = 0
    frame_num      = 0
    procesados     = 0

    modo_str = " [MODO TEST]" if modo_test else ""
    print(f"Video: {ancho_v}×{alto_v} @ {fps_orig:.1f} fps — {total} frames{modo_str}")
    print(f"frame_skip={FRAME_SKIP}  yolo_conf={YOLO_CONF_THRESH}")
    if modo_test:
        print("Modo test: solo visualizacion. Cerrar con 'q'.\n")
    else:
        print("Procesando... (presioná 'q' para cancelar)\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        mask_mog = sustractor.apply(frame)

        if frame_num % FRAME_SKIP != 0:
            continue
        procesados += 1

        pequeño     = cv2.resize(frame, (ancho_v // 2, alto_v // 2))
        pequeño_hsv = cv2.cvtColor(pequeño, cv2.COLOR_BGR2HSV)
        if not es_cenital(pequeño_hsv):
            continue
        frames_cenital += 1

        frame_yolo = cv2.resize(frame, (320, 180))
        yolo_boxes = detectar_con_yolo(frame_yolo, net)
        scale_x = frame.shape[1] / 320
        scale_y = frame.shape[0] / 180
        yolo_boxes = [(int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y), conf)
                      for x1, y1, x2, y2, conf in yolo_boxes]
        (torso_A_raw, zap_A_raw, bbox_A_raw, conf_A_raw,
         torso_B_raw, zap_B_raw, bbox_B_raw, conf_B_raw) = separar_jugadores_yolo(yolo_boxes)

        zap_A, ultima_A_p, frames_sin_A = filtrar_temporal(
            zap_A_raw, ultima_A_p, frames_sin_A)
        if zap_A is not None:
            torso_A, bbox_A, conf_A = torso_A_raw, bbox_A_raw, conf_A_raw
        else:
            torso_A = bbox_A = conf_A = None

        zap_B, ultima_B_p, frames_sin_B = filtrar_temporal(
            zap_B_raw, ultima_B_p, frames_sin_B)
        if zap_B is not None:
            torso_B, bbox_B, conf_B = torso_B_raw, bbox_B_raw, conf_B_raw
        else:
            torso_B = bbox_B = conf_B = None

        # ---- Dibujo común ----
        cv2.polylines(frame, [ZONA_TRACKING], isClosed=True,
                      color=(0, 0, 200), thickness=1, lineType=cv2.LINE_AA)
        cv2.polylines(frame, [CANCHA_PUNTOS], isClosed=True,
                      color=COLOR_CANCHA, thickness=1, lineType=cv2.LINE_AA)
        cv2.line(frame, RED_IZQ, RED_DER, (0, 140, 255), 2, cv2.LINE_AA)

        if modo_test:
            # Acumular zapatillas en cancha para trayectorias y heatmap
            for pos, calor, tray in [(zap_A, calor_A, tray_A_all),
                                     (zap_B, calor_B, tray_B_all)]:
                if pos is not None:
                    if cv2.pointPolygonTest(ZONA_TRACKING, (float(pos[0]), float(pos[1])), False) < 0:
                        continue
                    xi, yi = video_a_cancha(*pos, M_perspectiva)
                    xi = int(np.clip(xi, 0, ancho_c - 1))
                    yi = int(np.clip(yi, 0, alto_c - 1))
                    tray.append((xi, yi))
                    calor[yi, xi] += 1

            if procesados % 60 == 0:
                if zap_A:
                    xi, yi = video_a_cancha(*zap_A, M_perspectiva)
                    print(f"  Zap A video={zap_A} → cancha=({xi},{yi})  cancha size={ancho_c}x{alto_c}")
                if zap_B:
                    xi, yi = video_a_cancha(*zap_B, M_perspectiva)
                    print(f"  Zap B video={zap_B} → cancha=({xi},{yi})  cancha size={ancho_c}x{alto_c}")

            # Jugador A — bbox rosa + label conf + línea torso→zap + círculo blanco
            if zap_A is not None and bbox_A is not None:
                frames_A += 1
                x1, y1, x2, y2 = bbox_A
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BBOX_A, 2)
                cv2.putText(frame, f"A ({int(conf_A * 100)}%)",
                            (x1, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BBOX_A, 2, cv2.LINE_AA)
                cv2.line(frame, torso_A, zap_A, COLOR_BBOX_A, 2, cv2.LINE_AA)
                cv2.circle(frame, zap_A, 8, (255, 255, 255), -1)
            # Jugador B — bbox celeste + label conf + línea torso→zap + círculo blanco
            if zap_B is not None and bbox_B is not None:
                frames_B += 1
                x1, y1, x2, y2 = bbox_B
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BBOX_B, 2)
                cv2.putText(frame, f"B ({int(conf_B * 100)}%)",
                            (x1, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BBOX_B, 2, cv2.LINE_AA)
                cv2.line(frame, torso_B, zap_B, COLOR_BBOX_B, 2, cv2.LINE_AA)
                cv2.circle(frame, zap_B, 8, (255, 255, 255), -1)

            if procesados % 30 == 0:
                print(f"  Frame {frame_num:5d}  |  "
                      f"A: {frames_A:4d} det / {frames_cenital - frames_A:4d} no det  |  "
                      f"B: {frames_B:4d} det / {frames_cenital - frames_B:4d} no det")

        else:
            # Acumular zapatillas en cancha.png con perspectiva correcta
            def acumular(pos, calor, tray_all):
                if pos is None:
                    return 0
                if cv2.pointPolygonTest(ZONA_TRACKING, (float(pos[0]), float(pos[1])), False) < 0:
                    return 0
                xi, yi = video_a_cancha(*pos, M_perspectiva)
                xi = int(np.clip(xi, 0, ancho_c - 1))
                yi = int(np.clip(yi, 0, alto_c - 1))
                tray_all.append((xi, yi))
                calor[yi, xi] += 1
                return 1

            frames_A += acumular(zap_A, calor_A, tray_A_all)
            frames_B += acumular(zap_B, calor_B, tray_B_all)

            if procesados % 60 == 0:
                if zap_A:
                    xi, yi = video_a_cancha(*zap_A, M_perspectiva)
                    print(f"  Zap A video={zap_A} → cancha=({xi},{yi})  cancha size={ancho_c}x{alto_c}")
                if zap_B:
                    xi, yi = video_a_cancha(*zap_B, M_perspectiva)
                    print(f"  Zap B video={zap_B} → cancha=({xi},{yi})  cancha size={ancho_c}x{alto_c}")

            cola_A.append(zap_A)
            cola_B.append(zap_B)

            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            pelota    = detectar_pelota(frame_hsv, mascara_cancha, ultima_pelota, mask_mog)
            if pelota:
                ultima_pelota = pelota
            hist_pelota.append(pelota)
            cola_pelota.append(pelota)

            if detectar_golpe(hist_pelota, ultimo_golpe_p, procesados):
                golpes_count  += 1
                ultimo_golpe_p = procesados
                golpe_pos      = pelota or ultima_pelota

            dibujar_cola(frame, cola_A,      COLOR_A,      radio_max=4)
            dibujar_cola(frame, cola_B,      COLOR_B,      radio_max=4)
            dibujar_cola(frame, cola_pelota, COLOR_PELOTA, radio_max=3)

            dibujar_jugador(frame, torso_A, zap_A, COLOR_BBOX_A, "Jugador A")
            dibujar_jugador(frame, torso_B, zap_B, COLOR_BBOX_B, "Jugador B")

            if pelota:
                cv2.circle(frame, pelota, 9,  COLOR_PELOTA, -1)
                cv2.circle(frame, pelota, 11, (0, 0, 0),    1)

            frames_desde_golpe = procesados - ultimo_golpe_p
            if 0 < frames_desde_golpe <= 8 and golpe_pos and frames_desde_golpe % 2 == 1:
                cv2.circle(frame, golpe_pos, 28, COLOR_GOLPE, 3)
                cv2.putText(frame, "GOLPE", (golpe_pos[0] + 30, golpe_pos[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_GOLPE, 2, cv2.LINE_AA)

            dibujar_overlay(frame, frame_num, fps_orig, golpes_count, torso_A, torso_B)

            if procesados % 50 == 0:
                pct = 100 * frame_num // total
                print(f"  Frame {frame_num}/{total} ({pct}%) — "
                      f"cenital: {frames_cenital}  golpes: {golpes_count}")

        preview = cv2.resize(frame, (ancho_v // 3, alto_v // 3))
        cv2.imshow("Analisis Tenis", preview)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("  Cancelado por el usuario.")
            break

    cap.release()

    if not modo_test:
        print()
        print("=== RESUMEN DEL PARTIDO ===")
        print(f"Frames con vista cenital: {frames_cenital} / {procesados} procesados")
        print(f"Golpes detectados:         {golpes_count}")
        print(f"Jugador A detectado:       {frames_A} frames")
        print(f"Jugador B detectado:       {frames_B} frames")
        print("===========================")

    guardar_resumen_combinado(tray_A_all, tray_B_all, calor_A, calor_B,
                              img_cancha_rgb, frames_A, frames_B,
                              alto_c, rect_cancha_img)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0].startswith("-"):
        print("Uso: python analisis_tenis.py <video.mp4> [--test]")
        sys.exit(1)
    ruta   = args[0]
    test   = "--test" in args
    procesar_video(ruta, modo_test=test)
