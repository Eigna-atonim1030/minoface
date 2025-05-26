import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

def verificar_calidad_imagenes():
    """Verificar y filtrar imágenes de mala calidad"""
    dataPath = 'Data'
    total_removed = 0
    
    for person_name in os.listdir(dataPath):
        person_path = os.path.join(dataPath, person_name)
        if not os.path.isdir(person_path):
            continue
            
        print(f"🔍 Verificando calidad de imágenes para: {person_name}")
        removed_count = 0
        
        for image_file in os.listdir(person_path):
            if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(person_path, image_file)
                
                # Leer imagen
                img = cv2.imread(image_path, 0)
                if img is None:
                    os.remove(image_path)
                    removed_count += 1
                    continue
                
                # Verificar calidad (nitidez)
                blur_value = cv2.Laplacian(img, cv2.CV_64F).var()
                
                # Verificar tamaño
                if img.shape[0] < 100 or img.shape[1] < 100:
                    os.remove(image_path)
                    removed_count += 1
                    continue
                
                # Verificar que no esté muy borrosa
                if blur_value < 50:
                    os.remove(image_path)
                    removed_count += 1
                    continue
        
        remaining = len([f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"  ✅ {person_name}: {remaining} imágenes buenas, {removed_count} eliminadas")
        total_removed += removed_count
    
    print(f"🗑️ Total de imágenes de mala calidad eliminadas: {total_removed}")

def obtenerModelo():
    """Obtener datos de entrenamiento con validación"""
    dataPath = 'Data'
    peopleList = os.listdir(dataPath)
    print('🎯 Personas en base de datos:', peopleList)
    
    labels = []
    facesData = []
    label = 0
    
    for nameDir in peopleList:
        personPath = os.path.join(dataPath, nameDir)
        
        if not os.path.isdir(personPath):
            continue
            
        print(f'📖 Procesando imágenes de: {nameDir}')
        
        images = [f for f in os.listdir(personPath) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(images) == 0:
            print(f"⚠️ Advertencia: No hay imágenes en {personPath}")
            continue
        
        person_faces = []
        
        for fileName in images:
            img_path = os.path.join(personPath, fileName)
            img = cv2.imread(img_path, 0)
            
            if img is not None:
                # Redimensionar todas las imágenes al mismo tamaño
                img_resized = cv2.resize(img, (150, 150), interpolation=cv2.INTER_CUBIC)
                
                # Normalizar la imagen
                img_normalized = cv2.equalizeHist(img_resized)
                
                person_faces.append(img_normalized)
                labels.append(label)
        
        facesData.extend(person_faces)
        print(f"  ✅ {len(person_faces)} imágenes procesadas para {nameDir}")
        label += 1
    
    return facesData, labels

def entrenar_modelo_avanzado():
    """Entrenar modelo con validación cruzada"""
    print('🚀 ENTRENADOR AVANZADO DE RECONOCIMIENTO FACIAL')
    print('='*60)
    
    # Verificar calidad de imágenes primero
    verificar_calidad_imagenes()
    
    # Obtener datos
    faces, labels = obtenerModelo()
    
    if len(faces) == 0:
        print("❌ Error: No se encontraron imágenes válidas para entrenar")
        return
    
    # Estadísticas
    unique_labels = len(set(labels))
    print(f"📊 Estadísticas del dataset:")
    print(f"   • Total de imágenes: {len(faces)}")
    print(f"   • Número de personas: {unique_labels}")
    print(f"   • Promedio por persona: {len(faces)//unique_labels}")
    
    if len(faces) < 100:
        print("⚠️ ADVERTENCIA: Tienes pocas imágenes. Recomendado: 200+ por persona")
        response = input("¿Continuar de todos modos? (s/n): ")
        if response.lower() != 's':
            return
    
    # Dividir datos para validación
    X_train, X_test, y_train, y_test = train_test_split(
        faces, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"📚 Datos de entrenamiento: {len(X_train)} imágenes")
    print(f"🧪 Datos de prueba: {len(X_test)} imágenes")
    
    # Crear y entrenar reconocedor
    print("\n🤖 Entrenando modelo...")
    face_recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=2,          # Más preciso
        neighbors=16,      # Más vecinos para mejor precisión
        grid_x=8,         # Mayor resolución
        grid_y=8,
        threshold=80.0     # Umbral más alto para ser más estricto
    )
    
    face_recognizer.train(X_train, np.array(y_train))
    
    # Validar modelo
    print("🧪 Validando modelo...")
    correct_predictions = 0
    total_predictions = len(X_test)
    
    confidences = []
    
    for i, (test_face, true_label) in enumerate(zip(X_test, y_test)):
        predicted_label, confidence = face_recognizer.predict(test_face)
        confidences.append(confidence)
        
        if predicted_label == true_label:
            correct_predictions += 1
    
    accuracy = (correct_predictions / total_predictions) * 100
    avg_confidence = np.mean(confidences)
    
    print(f"📈 Resultados de validación:")
    print(f"   • Precisión: {accuracy:.1f}%")
    print(f"   • Confianza promedio: {avg_confidence:.1f}")
    print(f"   • Predicciones correctas: {correct_predictions}/{total_predictions}")
    
    # Determinar umbral recomendado
    if accuracy > 90:
        recommended_threshold = int(avg_confidence * 1.2)
        print(f"✅ Modelo excelente. Umbral recomendado: {recommended_threshold}")
    elif accuracy > 75:
        recommended_threshold = int(avg_confidence * 1.5)
        print(f"⚠️ Modelo aceptable. Umbral recomendado: {recommended_threshold}")
    else:
        recommended_threshold = int(avg_confidence * 2.0)
        print(f"❌ Modelo pobre. Necesitas más/mejores imágenes. Umbral: {recommended_threshold}")
    
    # Guardar modelo
    model_path = 'FacesModel.xml'
    face_recognizer.write(model_path)
    
    # Guardar configuración recomendada
    config_path = 'model_config.txt'
    with open(config_path, 'w') as f:
        f.write(f"recommended_threshold={recommended_threshold}\n")
        f.write(f"accuracy={accuracy:.1f}\n")
        f.write(f"avg_confidence={avg_confidence:.1f}\n")
        f.write(f"total_images={len(faces)}\n")
    
    print(f"\n✅ Modelo guardado como: {model_path}")
    print(f"⚙️ Configuración guardada en: {config_path}")
    print('🎉 Entrenamiento completado!')
    
    return recommended_threshold

if __name__ == "__main__":
    entrenar_modelo_avanzado()