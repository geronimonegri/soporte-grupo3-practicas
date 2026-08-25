import cv2
import numpy as np
from cv2 import aruco
import os


MARKER_SIZE = 0.05  # Tamano del marcador ArUco en metros (5 cm)

CALIB_FILE = "calibracion.npz"

# Tablero de ajedrez usado para calibrar: cantidad de ESQUINAS
# INTERNAS (no de casilleros). Un tablero de 10x7 casilleros
# tiene 9x6 esquinas internas.
CHESSBOARD_SIZE = (9, 6)
SQUARE_SIZE = 0.025      # Lado real de cada casillero, en metros (2.5 cm)
MIN_CALIB_SHOTS = 15     # Cantidad minima de capturas para calibrar bien

# ---------- Puntos 3D del marcador (plano, Z=0) ----------
marker_3d_points = np.array([
    [-MARKER_SIZE/2,  MARKER_SIZE/2, 0],
    [ MARKER_SIZE/2,  MARKER_SIZE/2, 0],
    [ MARKER_SIZE/2, -MARKER_SIZE/2, 0],
    [-MARKER_SIZE/2, -MARKER_SIZE/2, 0]
], dtype=np.float32)

# ---------- Los 8 vertices del cubo ----------
# Primeros 4: base (apoyada sobre el marcador, Z=0)
# Ultimos 4: techo (Z = MARKER_SIZE)
cube_3d_points = np.array([
    [-MARKER_SIZE/2, -MARKER_SIZE/2, 0],
    [ MARKER_SIZE/2, -MARKER_SIZE/2, 0],
    [ MARKER_SIZE/2,  MARKER_SIZE/2, 0],
    [-MARKER_SIZE/2,  MARKER_SIZE/2, 0],
    [-MARKER_SIZE/2, -MARKER_SIZE/2, MARKER_SIZE],
    [ MARKER_SIZE/2, -MARKER_SIZE/2, MARKER_SIZE],
    [ MARKER_SIZE/2,  MARKER_SIZE/2, MARKER_SIZE],
    [-MARKER_SIZE/2,  MARKER_SIZE/2, MARKER_SIZE]
], dtype=np.float32)

# Las 6 caras del cubo: cada una definida por 4 indices (en orden,
# recorriendo el contorno del cuadrado) + un color base en BGR.
CUBE_FACES = [
    ([0, 1, 2, 3], (50, 50, 200)),    # base    - rojo oscuro
    ([4, 5, 6, 7], (60, 200, 60)),    # techo   - verde
    ([0, 1, 5, 4], (200, 140, 40)),   # frente  - azul
    ([1, 2, 6, 5], (40, 170, 220)),   # derecha - naranja
    ([2, 3, 7, 6], (180, 60, 200)),   # atras   - violeta
    ([3, 0, 4, 7], (200, 200, 40)),   # izquierda - celeste/amarillo
]


def calibrar_camara():
    """
    Calibra la camara real usando un patron de tablero de ajedrez.
    Muestra la camara en vivo; cuando detecta el tablero lo marca en
    pantalla, y el usuario presiona 'c' para guardar esa captura.
    Con varias capturas desde angulos distintos, cv2.calibrateCamera
    calcula la matriz de camara (focal real, centro optico real) y
    los coeficientes de distorsion reales del lente, en vez de los
    valores estimados que se usaban en la version anterior.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: no se pudo abrir la camara para calibrar.")
        exit(1)

    # Puntos 3D del tablero en su propio sistema de referencia (Z=0).
    # mgrid arma la grilla de coordenadas (0,0), (1,0), (2,0)... y
    # luego se escala por el tamano real de cada casillero.
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = []  # puntos 3D reales, uno por captura
    imgpoints = []  # esquinas 2D detectadas, una por captura

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    print(f"Mostrale a la camara un tablero de ajedrez de {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]} esquinas internas.")
    print("Cuando el tablero se detecte (se van a marcar sus esquinas), presiona 'c' para capturar.")
    print(f"Se necesitan al menos {MIN_CALIB_SHOTS} capturas desde angulos y distancias distintas.")
    print("Presiona 'q' para cancelar la calibracion.")

    gray = None
    while len(objpoints) < MIN_CALIB_SHOTS:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

        display = frame.copy()
        if found:
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display, CHESSBOARD_SIZE, corners_refined, found)

        cv2.putText(display, f"Capturas: {len(objpoints)}/{MIN_CALIB_SHOTS}  ('c' capturar, 'q' cancelar)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Calibracion - tablero de ajedrez", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and found:
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            print(f"Captura {len(objpoints)}/{MIN_CALIB_SHOTS} guardada.")
        elif key == ord('q'):
            print("Calibracion cancelada por el usuario.")
            cap.release()
            cv2.destroyAllWindows()
            exit(1)

    cv2.destroyWindow("Calibracion - tablero de ajedrez")

    h, w = gray.shape[:2]
    print("Calculando calibracion...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (w, h), None, None
    )

    # Error de reproyeccion: que tan bien explican los parametros
    # calculados los puntos que realmente se vieron. Cuanto mas
    # cerca de 0, mejor la calibracion (valores tipicos: < 1 px).
    error_total = 0
    for i in range(len(objpoints)):
        proyectados, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error_total += cv2.norm(imgpoints[i], proyectados, cv2.NORM_L2) / len(proyectados)
    print(f"Error de reproyeccion promedio: {error_total / len(objpoints):.4f} px")

    np.savez(CALIB_FILE, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    print(f"Calibracion guardada en '{CALIB_FILE}' (la proxima vez se carga automaticamente).")

    cap.release()
    return camera_matrix.astype(np.float32), dist_coeffs.astype(np.float32)


def cargar_o_calibrar():
    """Si ya existe una calibracion guardada la reutiliza; si no, calibra de cero."""
    if os.path.exists(CALIB_FILE):
        data = np.load(CALIB_FILE)
        print(f"Calibracion cargada desde '{CALIB_FILE}'.")
        return data["camera_matrix"].astype(np.float32), data["dist_coeffs"].astype(np.float32)
    print("No se encontro una calibracion previa.")
    return calibrar_camara()


def dibujar_cubo_solido(frame, rvec, tvec, camera_matrix, dist_coeffs):
    """
    Proyecta el cubo 3D sobre el frame y dibuja sus 6 caras como
    poligonos SOLIDOS (en vez del wireframe de la version anterior).

    Algoritmo del pintor: se ordenan las caras de la mas lejana a la
    mas cercana (segun la profundidad Z de cada cara en el sistema de
    coordenadas de la CAMARA, no del marcador) y se dibujan en ese
    orden, para que las caras cercanas tapen correctamente a las que
    quedan detras.

    Ademas se aplica un sombreado simple: la normal de cada cara se
    compara con la direccion hacia la camara, y las caras mas de
    frente se dibujan mas brillantes que las caras de perfil, dando
    sensacion de volumen sin necesidad de texturas.
    """
    img_pts, _ = cv2.projectPoints(cube_3d_points, rvec, tvec, camera_matrix, dist_coeffs)
    img_pts = img_pts.reshape(-1, 2)

    # Matriz de rotacion 3x3 a partir del vector de rotacion (Rodrigues)
    R, _ = cv2.Rodrigues(rvec)
    # Vertices del cubo pasados al sistema de coordenadas de la camara
    verts_cam = (R @ cube_3d_points.T + tvec.reshape(3, 1)).T

    caras_ordenadas = sorted(
        CUBE_FACES,
        key=lambda cara: np.mean([verts_cam[i][2] for i in cara[0]]),
        reverse=True  # painter's algorithm: se dibuja primero lo mas lejos
    )

    overlay = frame.copy()
    for indices, color in caras_ordenadas:
        pts2d = img_pts[indices].astype(np.int32)

        # Normal de la cara en espacio camara, para el sombreado
        p0, p1, p2 = verts_cam[indices[0]], verts_cam[indices[1]], verts_cam[indices[2]]
        normal = np.cross(p1 - p0, p2 - p0)
        norma = np.linalg.norm(normal)
        if norma > 1e-9:
            normal = normal / norma

        # La camara mira hacia +Z, entonces las caras cuya normal
        # apunta hacia la camara (Z negativa) se ven "de frente" y
        # se iluminan mas; las de perfil quedan mas oscuras.
        brillo = 0.45 + 0.55 * max(0.0, -normal[2])
        color_sombreado = tuple(int(c * brillo) for c in color)

        cv2.fillConvexPoly(overlay, pts2d, color_sombreado)
        cv2.polylines(overlay, [pts2d], isClosed=True, color=(20, 20, 20), thickness=2)

    # Mezcla el overlay con el frame original (ALPHA < 1 deja ver
    # un poco el fondo debajo del cubo, sensacion de semitransparencia leve)
    ALPHA = 0.9
    cv2.addWeighted(overlay, ALPHA, frame, 1 - ALPHA, 0, dst=frame)


def main():
    camera_matrix, dist_coeffs = cargar_o_calibrar()
    print("Matriz de camara:\n", camera_matrix)
    print("Coeficientes de distorsion:\n", dist_coeffs)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la camara.")
        return

    # Configurar el detector ArUco
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    print("Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None:
            for i in range(len(ids)):
                img_points = corners[i][0]

                success, rvec, tvec = cv2.solvePnP(
                    marker_3d_points, img_points, camera_matrix, dist_coeffs
                )

                if success:
                    dibujar_cubo_solido(frame, rvec, tvec, camera_matrix, dist_coeffs)

        cv2.imshow('ArUco 3D - Cubo solido (camara calibrada)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
