import cv2
import os

# Crear carpeta Data si no existe
dataPath = 'Data'
if not os.path.exists(dataPath):
    os.makedirs(dataPath)

# Nombre de la persona (cambia esto por el nombre real)
personName = input("Ingresa el nombre de la persona: ")
personPath = os.path.join(dataPath, personName)

if not os.path.exists(personPath):
    os.makedirs(personPath)

# Usar cámara web local para capturar (más fácil para empezar)
cap = cv2.VideoCapture(0)  # Cámara web local
# Si quieres usar ESP32-CAM, cambia por: cap = cv2.VideoCapture('http://192.168.88.12:81/stream')

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara")
    exit()

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

count = 0
max_photos = 300

print(f"Capturando fotos de {personName}")
print("Presiona ESPACIO para capturar, ESC para salir")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()
    
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        
        # Capturar automáticamente cada 10 frames
        if count % 10 == 0:
            cv2.imwrite(os.path.join(personPath, f'rostro_{count}.jpg'), rostro)
            print(f"Foto {count}/{max_photos} capturada")
        
        count += 1
    
    cv2.putText(frame, f"Fotos: {count}/{max_photos}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Capturando rostros', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or count >= max_photos:  # ESC o máximo de fotos
        break

cap.release()
cv2.destroyAllWindows()
print(f"Captura completada. {count} fotos guardadas en {personPath}")