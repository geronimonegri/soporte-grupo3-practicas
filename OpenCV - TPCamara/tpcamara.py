import cv2

cap = cv2.VideoCapture(0)

grabando = False
out = None
contador = 0  

while True:
    _, frame = cap.read()

    # Botón azul FOTO
    cv2.circle(frame, (80, 430), 35, (255, 100, 0), -1)
    cv2.putText(frame, "FOTO", (58, 435), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # Botón GRABAR (círculo verde o rojo si está grabando)
    color_boton = (0, 0, 220) if grabando else (0, 200, 0)
    cv2.circle(frame, (200, 430), 35, color_boton, -1)
    cv2.putText(frame, "REC", (180, 435), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # Cuenta regresiva 
    if contador > 0:
        cv2.putText(frame, str(contador // 30 + 1), (290, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 8)
        contador -= 1

        if contador == 0:
            # Sacar la foto
            cv2.imwrite("foto.jpg", frame)

    # Grabar frame 
    if grabando:
        out.write(frame)

    cv2.imshow("Mi primer OpenCV", frame)

    tecla = cv2.waitKey(1) & 0xFF

    if tecla == ord('q'):
        break
    elif tecla == ord('f'):          # F = foto
        contador = 90                # 60 frames ≈ 3 segundos
    elif tecla == ord('r'):          # R = grabar / detener
        if not grabando:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter("video.avi", fourcc, 20.0, (640, 480))
            grabando = True
        else:
            out.release()
            grabando = False

cap.release()
cv2.destroyAllWindows()