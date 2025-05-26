import cv2
import os
import serial
import time
import serial.tools.list_ports

# Configuración del sistema
dataPath = 'Data'
imagePaths = os.listdir(dataPath) if os.path.exists(dataPath) else []

def find_arduino_port():
    """Encuentra automáticamente el puerto del Arduino"""
    print("🔍 Buscando Arduino...")
    ports = serial.tools.list_ports.comports()
    
    for port in ports:
        # Buscar puertos que pueden ser Arduino
        if any(keyword in port.description.upper() for keyword in ['ARDUINO', 'CH340', 'USB-SERIAL', 'FTDI']):
            print(f"✅ Posible Arduino encontrado en: {port.device} - {port.description}")
            return port.device
    
    # Si no encuentra automáticamente, mostrar puertos disponibles
    print("⚠️  Arduino no detectado automáticamente")
    print("Puertos disponibles:")
    for i, port in enumerate(ports):
        print(f"  {port.device} - {port.description}")
    
    return None

def setup_serial_connection():
    """Configurar conexión serial con Arduino"""
    arduino_port = find_arduino_port()
    
    if arduino_port is None:
        # Puertos comunes para probar
        common_ports = ['COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'COM10']
        print("🔄 Probando puertos comunes...")
        
        for port in common_ports:
            try:
                print(f"Probando {port}...")
                ser = serial.Serial(port, 9600, timeout=1)
                time.sleep(2)  # Esperar estabilización
                print(f"✅ Arduino conectado en: {port}")
                return ser
            except Exception as e:
                print(f"❌ {port} no disponible")
                continue
        
        print("❌ No se pudo conectar al Arduino automáticamente")
        manual_port = input("Ingresa el puerto manualmente (ej: COM4) o presiona Enter para continuar sin Arduino: ")
        
        if manual_port.strip():
            try:
                ser = serial.Serial(manual_port.strip(), 9600, timeout=1)
                time.sleep(2)
                print(f"✅ Arduino conectado manualmente en: {manual_port}")
                return ser
            except:
                print(f"❌ Error conectando a {manual_port}")
        
        return None
    else:
        try:
            ser = serial.Serial(arduino_port, 9600, timeout=1)
            time.sleep(2)
            return ser
        except Exception as e:
            print(f"❌ Error conectando a {arduino_port}: {e}")
            return None

def send_to_arduino(ser, result):
    """Enviar resultado al Arduino"""
    if ser is not None:
        try:
            if result == "DETECTADO":
                ser.write(b'1\n')  # Enviar '1' para rostro detectado
                print("📤 → Arduino: ROSTRO DETECTADO (Pantalla VERDE)")
            else:
                ser.write(b'0\n')  # Enviar '0' para rostro no detectado
                print("📤 → Arduino: ROSTRO NO DETECTADO (Pantalla ROJA)")
            ser.flush()  # Asegurar que se envíe
        except Exception as e:
            print(f"❌ Error enviando datos al Arduino: {e}")

def main():
    print("🚀 INICIANDO SISTEMA INTEGRADO")
    print("="*70)
    print("ESP32-CAM + Python + Arduino + Pantalla TFT")
    print("="*70)
    
    # Verificar modelo entrenado
    model_path = 'FacesModel.xml'
    if not os.path.exists(model_path):
        print("❌ Error: No se encuentra FacesModel.xml")
        print("Ejecuta primero: python TrainModel.py")
        return
    
    # Cargar configuración del modelo
    recommended_threshold = 3000  # Default
    try:
        with open('model_config.txt', 'r') as f:
            for line in f:
                if line.startswith('recommended_threshold='):
                    recommended_threshold = int(line.split('=')[1].strip())
                    break
        print(f"✅ Usando umbral recomendado: {recommended_threshold}")
    except:
        print(f"⚠️ Usando umbral por defecto: {recommended_threshold}")
    
    # Cargar modelo de reconocimiento
    try:
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.read(model_path)
        print("✅ Modelo de reconocimiento cargado")
        print(f"✅ Personas en base de datos: {imagePaths}")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return
    
    # Conectar a ESP32-CAM
    esp32_urls = [
        'http://192.168.88.12:81/stream',
        'http://192.168.88.12/stream',
        'http://192.168.88.12/',
    ]
    
    cap = None
    working_url = None
    
    print("📹 Conectando a ESP32-CAM...")
    for url in esp32_urls:
        print(f"Probando: {url}")
        test_cap = cv2.VideoCapture(url)
        if test_cap.isOpened():
            ret, frame = test_cap.read()
            if ret and frame is not None:
                print(f"✅ ESP32-CAM conectado: {url}")
                cap = test_cap
                working_url = url
                break
            test_cap.release()
    
    if cap is None:
        print("❌ Error: No se pudo conectar al ESP32-CAM")
        print("Verifica que esté encendido y en la red WiFi")
        return
    
    # Conectar a Arduino
    print("\n🔌 Configurando conexión con Arduino...")
    arduino_serial = setup_serial_connection()
    
    if arduino_serial:
        print("✅ Arduino conectado - Pantalla TFT lista")
        # Enviar comando inicial para mostrar pantalla de espera
        send_to_arduino(arduino_serial, "NO_DETECTADO")
    else:
        print("⚠️  Continuando sin Arduino (solo reconocimiento en PC)")
    
    # Cargar detector de rostros
    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    print("\n" + "="*70)
    print("🎥 SISTEMA DE RECONOCIMIENTO ACTIVO")
    print("="*70)
    print("• ESP32-CAM: ✅ Streaming activo")
    print("• Python: ✅ Reconocimiento facial")
    print(f"• Arduino: {'✅ Pantalla TFT activa' if arduino_serial else '❌ Sin conexión'}")
    print("• Presiona 'q' para salir")
    print("="*70)
    
    last_result = None
    stable_count = 0
    required_stability = 10  # CAMBIADO: Era 5, ahora 10 frames para más estabilidad
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error capturando video desde ESP32-CAM")
            break
        
        # VOLTEAR LA IMAGEN SI ESTÁ AL REVÉS
        # Opciones de rotación/volteo:
        frame = cv2.flip(frame, -1)  # Voltear horizontal y vertical (180°)
        # frame = cv2.flip(frame, 0)   # Solo voltear vertical
        # frame = cv2.flip(frame, 1)   # Solo voltear horizontal
        # frame = cv2.rotate(frame, cv2.ROTATE_180)  # Rotar 180°
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar rostros - PARÁMETROS MUY ESTRICTOS
        faces = faceClassif.detectMultiScale(
            gray, 
            scaleFactor=1.4,      # CAMBIADO: Menos sensible (era 1.3)
            minNeighbors=8,       # CAMBIADO: Más estricto (era 6)
            minSize=(120, 120),   # CAMBIADO: Rostros más grandes (era 100)
            maxSize=(250, 250)    # CAMBIADO: Rango más pequeño (era 300)
        )
        
        current_result = "NO_DETECTADO"
        best_confidence = float('inf')
        detected_person = "Desconocido"
        
        for (x, y, w, h) in faces:
            # Extraer rostro
            rostro = gray[y:y + h, x:x + w] 
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            
            # Reconocimiento facial
            result = face_recognizer.predict(rostro)
            confidence = result[1]
            predicted_person = result[0]
            
            # USAR UMBRAL DINÁMICO CALCULADO POR EL ENTRENADOR
            if confidence < recommended_threshold and predicted_person < len(imagePaths):
                # ROSTRO AUTORIZADO
                person_name = imagePaths[predicted_person]
                current_result = "DETECTADO"
                detected_person = person_name
                best_confidence = confidence
                
                cv2.putText(frame, 'ROSTRO DETECTADO', (x, y - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                cv2.putText(frame, f'Usuario: {person_name}', (x, y + h + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f'Confianza: {confidence:.0f}', (x, y + h + 55), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
                
            else:
                # ROSTRO NO AUTORIZADO
                cv2.putText(frame, 'ROSTRO NO DETECTADO', (x, y - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                cv2.putText(frame, 'ACCESO DENEGADO', (x, y + h + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f'Confianza: {confidence:.0f}', (x, y + h + 55), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)
        
        # Sistema de estabilidad para evitar parpadeo
        if current_result == last_result:
            stable_count += 1
        else:
            stable_count = 0
            last_result = current_result
        
        # Enviar al Arduino solo cuando el resultado sea estable
        if stable_count == required_stability:
            send_to_arduino(arduino_serial, current_result)
            if current_result == "DETECTADO":
                print(f"✅ ACCESO AUTORIZADO - {detected_person} (Confianza: {best_confidence:.0f})")
            else:
                print(f"❌ ACCESO DENEGADO - Sin rostros autorizados")
        
        # Mostrar información del sistema
        status_color = (0, 255, 0) if len(faces) > 0 else (255, 255, 255)
        cv2.putText(frame, f'Sistema Integrado - Rostros: {len(faces)}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Mostrar estado de conexiones
        arduino_status = "ON" if arduino_serial else "OFF"
        arduino_color = (0, 255, 0) if arduino_serial else (0, 0, 255)
        cv2.putText(frame, f'Arduino TFT: {arduino_status}', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, arduino_color, 2)
        
        cv2.putText(frame, f'ESP32-CAM: {working_url}', (10, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Mostrar video
        cv2.imshow('🎥 Sistema Integrado ESP32-CAM + Arduino TFT', frame)
        
        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Limpiar recursos
    cap.release()
    cv2.destroyAllWindows()
    if arduino_serial:
        arduino_serial.close()
    
    print("\n🛑 Sistema detenido correctamente")
    print("="*70)

if __name__ == "__main__":
    main()