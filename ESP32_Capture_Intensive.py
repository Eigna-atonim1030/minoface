import cv2
import os

def main():
    # Crear carpeta Data si no existe
    dataPath = 'Data'
    if not os.path.exists(dataPath):
        os.makedirs(dataPath)
    
    # Nombre de la persona
    personName = input("Ingresa el nombre de la persona (o presiona Enter para 'nicol'): ")
    if not personName.strip():
        personName = 'nicol'
    
    personPath = os.path.join(dataPath, personName)
    
    # Contar fotos existentes y preguntar si agregar más
    if os.path.exists(personPath):
        existing_photos = len([f for f in os.listdir(personPath) if f.endswith('.jpg')])
        print(f"📁 Fotos existentes: {existing_photos}")
        if existing_photos > 0:
            response = input(f"¿Agregar más fotos a las {existing_photos} existentes? (s/n): ")
            if response.lower() != 's':
                print("❌ Captura cancelada")
                return
    else:
        os.makedirs(personPath)
        existing_photos = 0
    
    # CONECTAR A ESP32-CAM - Probar múltiples URLs
    esp32_urls = [
        'http://192.168.88.12:81/stream',
        'http://192.168.88.12/stream',
        'http://192.168.88.12:81/',
        'http://192.168.88.12/',
        'http://192.168.88.12/capture'
    ]
    
    cap = None
    working_url = None
    
    print("🔍 Buscando ESP32-CAM...")
    for url in esp32_urls:
        print(f"Probando: {url}")
        test_cap = cv2.VideoCapture(url)
        if test_cap.isOpened():
            ret, frame = test_cap.read()
            if ret and frame is not None:
                print(f"✅ ¡ESP32-CAM encontrada! URL: {url}")
                cap = test_cap
                working_url = url
                break
            else:
                test_cap.release()
        else:
            test_cap.release()
    
    # Si no encuentra ESP32-CAM, usar cámara local
    if cap is None:
        print("❌ No se encontró ESP32-CAM")
        print("🔄 Usando cámara web local...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: No se pudo abrir ninguna cámara")
            return
        working_url = "cámara local"
    
    print(f"📹 Conectado a: {working_url}")
    
    # Cargar detector de rostros
    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # CONFIGURACIÓN PARA ENTRENAMIENTO ROBUSTO
    count = 0
    saved_photos = 0
    max_photos = 3000  # MÁS frames para procesar
    target_photos = 500  # META: 500 fotos de ALTA CALIDAD
    start_number = existing_photos
    
    print("\n" + "="*70)
    print("🚀 CAPTURA MASIVA DESDE ESP32-CAM")
    print("="*70)
    print(f"👤 Persona: {personName}")
    print(f"📁 Carpeta: {personPath}")
    print(f"📊 Fotos existentes: {existing_photos}")
    print(f"🎯 Meta total: {target_photos} fotos")
    print(f"🎥 Fuente: {working_url}")
    print("\n📋 INSTRUCCIONES PARA 1000 FOTOS:")
    print("• 🔄 MUÉVETE CONSTANTEMENTE - cabeza, expresiones, distancia")
    print("• 😀😐😮 Cambia expresiones cada 3 segundos")
    print("• 👈👉⬆️⬇️ Gira e inclina la cabeza continuamente")
    print("• 📏 Acércate y aléjate de la ESP32-CAM")
    print("• ⚡ Captura cada 3 frames = MUY RÁPIDO")
    print("• 🛑 Presiona ESC para terminar")
    print("="*70)
    
    # Parámetros de captura para CALIDAD no cantidad
    frames_between_saves = 8  # CAMBIADO: Cada 8 frames (era 3) - más selectivo
    min_blur_threshold = 80   # CAMBIADO: Más estricto con calidad (era 30)
    
    while count < max_photos and saved_photos < (target_photos - existing_photos):
        ret, frame = cap.read()
        if not ret:
            print("❌ Error capturando video desde ESP32-CAM")
            break
        
        # VOLTEAR LA IMAGEN SI ESTÁ AL REVÉS
        frame = cv2.flip(frame, -1)  # Voltear horizontal y vertical (180°)
        # Si necesitas otra rotación, usa una de estas:
        # frame = cv2.flip(frame, 0)   # Solo voltear vertical
        # frame = cv2.flip(frame, 1)   # Solo voltear horizontal
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = gray.copy()
        
        # Detectar rostros - MÁS PERMISIVO
        faces = faceClassif.detectMultiScale(
            gray, 
            scaleFactor=1.1,    # Más sensible
            minNeighbors=3,     # Menos estricto
            minSize=(60, 60),   # Rostros más pequeños
            maxSize=(400, 400)
        )
        
        # Procesar cada rostro detectado
        for (x, y, w, h) in faces:
            # Extraer rostro
            rostro = auxFrame[y:y + h, x:x + w]
            
            # Verificar calidad básica
            blur_value = cv2.Laplacian(rostro, cv2.CV_64F).var()
            
            # GUARDAR CADA POCOS FRAMES con calidad mínima
            if blur_value > min_blur_threshold and count % frames_between_saves == 0:
                # Redimensionar a tamaño estándar
                rostro_resized = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
                
                # Guardar foto
                filename = f'rostro_{start_number + saved_photos:04d}.jpg'
                filepath = os.path.join(personPath, filename)
                cv2.imwrite(filepath, rostro_resized)
                saved_photos += 1
                
                # Mostrar progreso cada 25 fotos
                if saved_photos % 25 == 0:
                    total_current = existing_photos + saved_photos
                    remaining = target_photos - total_current
                    progress = (total_current * 100) // target_photos
                    print(f"📸 {total_current}/{target_photos} ({progress}%) | Faltan: {remaining} | Calidad: {blur_value:.0f}")
            
            # Dibujar rectángulo en el video
            color = (0, 255, 0) if blur_value > min_blur_threshold else (0, 165, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Mostrar calidad
            cv2.putText(frame, f'Q: {blur_value:.0f}', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Información en pantalla
        total_current = existing_photos + saved_photos
        progress = (total_current * 100) // target_photos
        remaining = target_photos - total_current
        
        cv2.putText(frame, f'TOTAL: {total_current}/{target_photos} ({progress}%)', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f'NUEVAS: {saved_photos} | FALTAN: {remaining}', 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, 'MUEVETE CONSTANTEMENTE!', 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.putText(frame, f'Fuente: {working_url}', 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Mostrar video
        cv2.imshow(f'ESP32-CAM Captura Masiva - {personName}', frame)
        
        # Control de teclado
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("🛑 Captura detenida por el usuario")
            break
        
        count += 1
        
        # Salir si alcanzamos la meta
        if (existing_photos + saved_photos) >= target_photos:
            print("🎯 ¡META DE 1000 FOTOS ALCANZADA!")
            break
    
    # Limpiar recursos
    cap.release()
    cv2.destroyAllWindows()
    
    # Resumen final
    total_final = existing_photos + saved_photos
    print("\n" + "="*70)
    print("🎉 CAPTURA DESDE ESP32-CAM COMPLETADA")
    print("="*70)
    print(f"📸 Fotos nuevas capturadas: {saved_photos}")
    print(f"📁 TOTAL EN CARPETA: {total_final}")
    print(f"📂 Ubicación: {personPath}")
    print(f"🎥 Fuente utilizada: {working_url}")
    
    if total_final >= 1000:
        print("🏆 ¡INCREÍBLE! 1000+ fotos - Entrenamiento de ÉLITE garantizado")
    elif total_final >= 800:
        print("🥇 ¡EXCELENTE! Entrenamiento muy robusto")
    elif total_final >= 500:
        print("🥈 ¡MUY BIEN! Entrenamiento sólido")
    elif total_final >= 200:
        print("🥉 Buena cantidad - Funcionará bien")
    else:
        print("⚠️  Ejecuta nuevamente para más fotos")
    
    print("\n📋 PRÓXIMOS PASOS:")
    print("   1. python TrainModel.py")
    print("   2. python FaceRecognition.py")
    print("="*70)

if __name__ == "__main__":
    main()