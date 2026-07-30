#VIDEO https://drive.google.com/file/d/1SzZOUE2RoGyOJsnPjHn1BvEOWGsL2bzO/view
import cv2
import numpy as np


def distancia(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def trackear_autos(video_path, ancho=800):
    cap = cv2.VideoCapture(video_path)
    ret, prueba = cap.read()
    if not ret:
        print(f"No se pudo abrir el video: {video_path}")
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    h_orig, w_orig = prueba.shape[:2]
    alto = int(h_orig * ancho / w_orig)

    sustractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=50, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # autos: id -> {pos, vel, frames_visto, frames_desaparecido, contado, cruzo}
    autos = {}
    siguiente_id = 1          # arranca en 1, no en 0
    total_contados = 0

    FRAMES_CONFIRMACION = 6
    FRAMES_TOLERANCIA   = 60
    MAX_DISTANCIA       = 150

    # Línea horizontal a mitad de pantalla, más arriba para interceptar autos a tiempo
    LINEA_Y = int(alto * 0.50)

    # Solo trackear carriles derechos: 60% del ancho hacia la derecha
    CARRIL_X = int(ancho * 0.60)

    # Duración máxima: 15 segundos
    DURACION_FRAMES = int(fps * 15)
    frame_actual = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_actual >= DURACION_FRAMES:
                break

            frame = cv2.resize(frame, (ancho, alto))

            mascara = sustractor.apply(frame)
            mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN,  kernel)
            mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
            mascara = cv2.dilate(mascara, kernel, iterations=3)

            contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            centroides_actuales = []
            for cnt in contornos:
                if cv2.contourArea(cnt) < 400:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                cx, cy = x + w // 2, y + h // 2
                # Solo considerar detecciones en la mitad derecha de la pantalla
                if cx < CARRIL_X:
                    continue
                centroides_actuales.append((cx, cy, x, y, w, h))

            ids_usados = set()
            centroides_no_asignados = []

            for (cx, cy, x, y, w, h) in centroides_actuales:
                mejor_id   = None
                mejor_dist = MAX_DISTANCIA

                for pid, info in autos.items():
                    if pid in ids_usados:
                        continue
                    px = info["pos"][0] + info["vel"][0]
                    py = info["pos"][1] + info["vel"][1]
                    d = distancia((cx, cy), (px, py))
                    if d < mejor_dist:
                        mejor_dist = d
                        mejor_id   = pid

                if mejor_id is not None:
                    pos_ant = autos[mejor_id]["pos"]
                    vx = 0.5 * (cx - pos_ant[0]) + 0.5 * autos[mejor_id]["vel"][0]
                    vy = 0.5 * (cy - pos_ant[1]) + 0.5 * autos[mejor_id]["vel"][1]
                    autos[mejor_id]["pos"] = (cx, cy)
                    autos[mejor_id]["vel"] = (vx, vy)
                    autos[mejor_id]["frames_visto"] += 1
                    autos[mejor_id]["frames_desaparecido"] = 0
                    ids_usados.add(mejor_id)

                    if (autos[mejor_id]["frames_visto"] == FRAMES_CONFIRMACION
                            and not autos[mejor_id]["contado"]):
                        autos[mejor_id]["contado"] = True

                    # Contar si: confirmado + venía de arriba de la línea + ahora está abajo
                    if (autos[mejor_id]["contado"]
                            and not autos[mejor_id]["cruzo"]
                            and pos_ant[1] <= LINEA_Y < cy):        # cruzó la línea este frame
                        autos[mejor_id]["cruzo"] = True
                        total_contados += 1

                    if autos[mejor_id]["contado"]:
                        color = (0, 255, 0) if autos[mejor_id]["cruzo"] else (0, 200, 255)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                        cv2.putText(frame, f"Auto {mejor_id}", (x, y - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                else:
                    centroides_no_asignados.append((cx, cy, x, y, w, h))

            for (cx, cy, x, y, w, h) in centroides_no_asignados:
                autos[siguiente_id] = {
                    "pos": (cx, cy),
                    "vel": (0, 0),
                    "frames_visto": 1,
                    "frames_desaparecido": 0,
                    "contado": False,
                    "cruzo": False
                }
                siguiente_id += 1

            eliminar = []
            for pid, info in autos.items():
                if pid not in ids_usados:
                    autos[pid]["frames_desaparecido"] += 1
                    if autos[pid]["frames_desaparecido"] > FRAMES_TOLERANCIA:
                        eliminar.append(pid)
                    elif info["contado"]:
                        cx, cy = int(info["pos"][0]), int(info["pos"][1])
                        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                        cv2.putText(frame, f"Auto {pid}", (cx - 20, cy - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)
            for pid in eliminar:
                del autos[pid]

            visibles = sum(1 for info in autos.values()
                           if info["contado"] and info["frames_desaparecido"] < FRAMES_TOLERANCIA)

            # Línea horizontal fina (línea de meta) - de punta a punta
            cv2.line(frame, (0, LINEA_Y), (ancho, LINEA_Y), (0, 0, 255), 1)

            segundos_restantes = max(0, (DURACION_FRAMES - frame_actual) / fps)

            cv2.putText(frame, f"En pantalla: {visibles}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"Total contados: {total_contados}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Tiempo: {segundos_restantes:.1f}s", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 180, 0), 2)

            cv2.imshow("Tracking de autos", frame)
            frame_actual += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\nResultado final: {total_contados} autos cruzaron la linea en 15 segundos\n")


if __name__ == "__main__":
    trackear_autos("12912796_2160_3840_30fps.mp4")

