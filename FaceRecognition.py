import cv2
import os

# Usar ruta relativa
dataPath = 'Data'
if not os.path.exists(dataPath):
    print(f"Error: La carpeta {dataPath} no existe")
    exit()

imagePaths = os.listdir(dataPath)

def main():
    print('Personas en base de datos:', imagePaths)

    # Verificar si existe el modelo entrenado
    model_path = 'FacesModel.xml'
    if not os.path.exists(model_path):
        print(f"Error: No se encuentra el archivo {model_path}")
        print("Primero ejecuta TrainModel.py para entrenar el modelo")
        return

    try:
        # Crear el reconocedor de caras y cargar el modelo preentrenado
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.read(model_path)
        print("Modelo cargado exitosamente")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return

    # Conectar a ESP32-CAM
    esp32_urls = [
        'http://192.168.88.12:81/stream',
        'http://192.168.88.12/stream', 
        'http://192.168.88.12/capture'
    ]
    
    cap = None
    working_url = None
    
    print("Probando conexiones al ESP32-CAM...")
    for url in esp32_urls:
        print(f"Probando: {url}")
        test_cap = cv2.VideoCapture(url)
        if test_cap.isOpened():
            ret, frame = test_cap.read()
            if ret and frame is not None:
                print(f"✓ Conexión exitosa: {url}")
                cap = test_cap
                working_url = url
                break
            else:
                test_cap.release()
        else:
            test_cap.release()
    
    if cap is None:
        print("Error: No se pudo conectar al ESP32-CAM")
        return

    print(f"Conectado exitosamente a: {working_url}")
    print("=== SISTEMA DE SEGURIDAD FACIAL ===")
    print("Presiona 'q' para salir")
    print("Solo usuarios autorizados tendrán acceso")
    print("=====================================")
    
    # Cargar el clasificador Haar Cascade para detección de rostros
    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo capturar el video.")
            break

        # Convertir la imagen a escala de grises para la detección de rostros
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar rostros
        faces = faceClassif.detectMultiScale(
            gray, 
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(50, 50)
        )

        print(f"DEBUG: Rostros detectados: {len(faces)}")  # Debug en consola

        for (x, y, w, h) in faces:
            # Extraer el rostro detectado
            rostro = gray[y:y + h, x:x + w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)

            # Predecir el rostro detectado usando el modelo entrenado
            result = face_recognizer.predict(rostro)
            confidence = result[1]
            predicted_person = result[0]
            
            print(f"DEBUG: Confianza: {confidence:.1f}, Persona: {predicted_person}")
            
            # LÓGICA PRINCIPAL: Determinar si es autorizado o no
            # Umbral ajustado para ser más preciso
            if confidence < 7000 and predicted_person < len(imagePaths):
                # ES LA PERSONA AUTORIZADA (nicol)
                person_name = imagePaths[predicted_person]
                
                # TEXTO Y RECTANGULO VERDE
                cv2.putText(frame, 'ROSTRO DETECTADO', (x, y - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, f'Usuario: {person_name}', (x, y + h + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
                
                print(f"✓ ACCESO AUTORIZADO - {person_name} (Confianza: {confidence:.1f})")
            else:
                # NO ES LA PERSONA AUTORIZADA
                
                # TEXTO Y RECTANGULO ROJO
                cv2.putText(frame, 'ROSTRO NO DETECTADO', (x, y - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.putText(frame, 'ACCESO DENEGADO', (x, y + h + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)
                
                print(f"✗ ACCESO DENEGADO - Persona no autorizada (Confianza: {confidence:.1f})")

        # Mostrar información del sistema en la esquina
        cv2.putText(frame, f'Sistema de Seguridad - Rostros: {len(faces)}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Mostrar el video
        cv2.imshow('Sistema de Seguridad ESP32-CAM', frame)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()