import face_recognition
import face_recognition_models
import os
import cv2
import serial
import time
import serial.tools.list_ports
import numpy as np
import pickle

# Sobrescribimos la ruta manualmente para que face_recognition los encuentre
face_recognition_models.models = os.path.join(
    os.path.dirname(face_recognition_models.__file__), 'models'
)

class DeepFaceRecognitionSystem:
    def __init__(self, data_path='Data'):
        self.data_path = data_path
        self.known_face_encodings = []
        self.known_face_names = []
        self.encodings_file = 'face_encodings.pkl'
        
    def load_or_create_encodings(self):
        """Cargar encodings existentes o crear nuevos"""
        if os.path.exists(self.encodings_file):
            print("📁 Cargando encodings existentes...")
            try:
                with open(self.encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data['encodings']
                    self.known_face_names = data['names']
                print(f"✅ Cargados {len(self.known_face_encodings)} encodings")
                return True
            except:
                print("❌ Error cargando encodings, creando nuevos...")
                
        return self.create_face_encodings()
    
    def create_face_encodings(self):
        """Crear encodings faciales usando deep learning"""
        print("🧠 Creando encodings con Deep Learning...")
        print("⏳ Esto puede tomar varios minutos...")
        
        if not os.path.exists(self.data_path):
            print(f"❌ No existe la carpeta {self.data_path}")
            return False
            
        total_processed = 0
        
        for person_name in os.listdir(self.data_path):
            person_path = os.path.join(self.data_path, person_name)
            
            if not os.path.isdir(person_path):
                continue
                
            print(f"👤 Procesando: {person_name}")
            person_encodings = []
            
            image_files = [f for f in os.listdir(person_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for i, image_file in enumerate(image_files):
                if i % 20 == 0:  # Mostrar progreso cada 20 imágenes
                    print(f"  📸 Procesando imagen {i+1}/{len(image_files)}")
                    
                image_path = os.path.join(person_path, image_file)
                
                # Cargar imagen
                image = face_recognition.load_image_file(image_path)
                
                # Obtener encodings (128 dimensiones)
                face_encodings = face_recognition.face_encodings(image)
                
                if face_encodings:
                    # Tomar solo el primer rostro detectado
                    encoding = face_encodings[0]
                    person_encodings.append(encoding)
                    total_processed += 1
            
            # Agregar encodings de esta persona
            self.known_face_encodings.extend(person_encodings)
            self.known_face_names.extend([person_name] * len(person_encodings))
            
            print(f"  ✅ {len(person_encodings)} encodings creados para {person_name}")
        
        if total_processed > 0:
            # Guardar encodings para uso futuro
            data = {
                'encodings': self.known_face_encodings,
                'names': self.known_face_names
            }
            with open(self.encodings_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"💾 Encodings guardados en {self.encodings_file}")
            print(f"🎯 Total: {total_processed} encodings creados")
            return True
        else:
            print("❌ No se pudieron crear encodings")
            return False
    
    def recognize_faces(self, frame):
        """Reconocer rostros usando deep learning"""
        # Redimensionar frame para procesamiento más rápido
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Encontrar rostros y encodings
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        results = []
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Escalar coordenadas de vuelta al tamaño original
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # Comparar con rostros conocidos
            matches = face_recognition.compare_faces(
                self.known_face_encodings, 
                face_encoding, 
                tolerance=0.4  # MÁS ESTRICTO (default: 0.6)
            )
            
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            
            name = "DESCONOCIDO"
            confidence = 0
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                
                # VERIFICACIÓN DOBLE: match Y distancia
                if matches[best_match_index] and face_distances[best_match_index] < 0.4:
                    name = self.known_face_names[best_match_index]
                    confidence = 1 - face_distances[best_match_index]
            
            results.append({
                'location': (left, top, right, bottom),
                'name': name,
                'confidence': confidence,
                'distance': face_distances[best_match_index] if len(face_distances) > 0 else 1.0
            })
        
        return results

def find_arduino_port():
    """Encuentra automáticamente el puerto del Arduino"""
    print("🔍 Buscando Arduino...")
    ports = serial.tools.list_ports.comports()
    
    for port in ports:
        if any(keyword in port.description.upper() for keyword in ['ARDUINO', 'CH340', 'USB-SERIAL', 'FTDI']):
            print(f"✅ Arduino encontrado: {port.device}")
            return port.device
    
    print("⚠️ Arduino no detectado automáticamente")
    return None

def setup_serial_connection():
    """Configurar conexión con Arduino"""
    arduino_port = find_arduino_port()
    
    if arduino_port is None:
        common_ports = ['COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8']
        for port in common_ports:
            try:
                ser = serial.Serial(port, 9600, timeout=1)
                time.sleep(2)
                print(f"✅ Arduino conectado en: {port}")
                return ser
            except:
                continue
        return None
    else:
        try:
            ser = serial.Serial(arduino_port, 9600, timeout=1)
            time.sleep(2)
            return ser
        except:
            return None

def send_to_arduino(ser, result):
    """Enviar resultado al Arduino"""
    if ser is not None:
        try:
            if result == "DETECTADO":
                ser.write(b'1\n')
                print("📤 → Arduino: ROSTRO DETECTADO (Pantalla VERDE)")
            else:
                ser.write(b'0\n')
                print("📤 → Arduino: ROSTRO NO DETECTADO (Pantalla ROJA)")
            ser.flush()
        except Exception as e:
            print(f"❌ Error enviando al Arduino: {e}")

def main():
    print("🧠 SISTEMA DE RECONOCIMIENTO FACIAL CON DEEP LEARNING")
    print("="*70)
    
    # Inicializar sistema de reconocimiento
    face_system = DeepFaceRecognitionSystem()
    
    # Cargar o crear encodings
    if not face_system.load_or_create_encodings():
        print("❌ Error: No se pudieron cargar/crear los encodings faciales")
        return
    
    # Conectar a ESP32-CAM
    esp32_urls = [
        'http://192.168.88.12:81/stream',
        'http://192.168.88.12/stream',
        'http://192.168.88.12/'
    ]
    
    cap = None
    working_url = None
    
    print("\n📹 Conectando a ESP32-CAM...")
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
        print("❌ No se pudo conectar al ESP32-CAM")
        return
    
    # Conectar Arduino
    print("\n🔌 Conectando Arduino...")
    arduino_serial = setup_serial_connection()
    
    if arduino_serial:
        print("✅ Arduino conectado - Pantalla TFT lista")
    else:
        print("⚠️ Continuando sin Arduino")
    
    print("\n" + "="*70)
    print("🎥 SISTEMA DE DEEP LEARNING ACTIVO")
    print("="*70)
    print("• 🧠 Deep Learning: ✅ Reconocimiento de última generación")
    print("• 📹 ESP32-CAM: ✅ Streaming activo") 
    print(f"• 🔌 Arduino: {'✅ Conectado' if arduino_serial else '❌ Desconectado'}")
    print("• ⌨️ Presiona 'q' para salir")
    print("="*70)
    
    frame_count = 0
    last_result = None
    stable_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error capturando video")
            break
        
        # Voltear imagen si está al revés
        frame = cv2.flip(frame, -1)
        
        frame_count += 1
        
        # Procesar cada 5 frames para mejor rendimiento
        if frame_count % 5 == 0:
            # Reconocer rostros con deep learning
            face_results = face_system.recognize_faces(frame)
            
            current_result = "NO_DETECTADO"
            
            for result in face_results:
                left, top, right, bottom = result['location']
                name = result['name']
                confidence = result['confidence']
                distance = result['distance']
                
                if name != "DESCONOCIDO":
                    # ROSTRO AUTORIZADO DETECTADO
                    current_result = "DETECTADO"
                    
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
                    cv2.putText(frame, 'ROSTRO DETECTADO', (left, top - 35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, f'Usuario: {name}', (left, bottom + 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, f'Confianza: {confidence:.1%}', (left, bottom + 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    print(f"✅ ACCESO AUTORIZADO - {name} (Confianza: {confidence:.1%}, Distancia: {distance:.3f})")
                else:
                    # ROSTRO NO AUTORIZADO
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)
                    cv2.putText(frame, 'ROSTRO NO DETECTADO', (left, top - 35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(frame, 'ACCESO DENEGADO', (left, bottom + 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(frame, f'Distancia: {distance:.3f}', (left, bottom + 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
                    print(f"❌ ACCESO DENEGADO - Persona no autorizada (Distancia: {distance:.3f})")
            
            # Sistema de estabilidad
            if current_result == last_result:
                stable_count += 1
            else:
                stable_count = 0
                last_result = current_result
            
            # Enviar al Arduino cuando sea estable
            if stable_count == 3:  # 3 detecciones consecutivas
                send_to_arduino(arduino_serial, current_result)
        
        # Mostrar información del sistema
        cv2.putText(frame, 'Deep Learning Face Recognition', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Frame: {frame_count}', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Mostrar video
        cv2.imshow('🧠 Deep Learning Face Recognition System', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Limpiar recursos
    cap.release()
    cv2.destroyAllWindows()
    if arduino_serial:
        arduino_serial.close()
    
    print("\n🛑 Sistema detenido")
    print("="*70)

if __name__ == "__main__":
    main()