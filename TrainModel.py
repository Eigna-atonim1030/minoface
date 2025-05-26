import cv2
import os
import numpy as np

def obtenerModelo():
    dataPath = 'Data'  # Ruta relativa corregida
    peopleList = os.listdir(dataPath)
    print('Lista de personas: ', peopleList)
    
    labels = []
    facesData = []
    label = 0
    
    for nameDir in peopleList:
        personPath = os.path.join(dataPath, nameDir)
        
        # Verificar si es un directorio
        if not os.path.isdir(personPath):
            continue
            
        print('Leyendo las imágenes de:', nameDir)
        
        images = os.listdir(personPath)
        if len(images) == 0:
            print(f"Advertencia: No hay imágenes en {personPath}")
            continue
        
        for fileName in images:
            if fileName.lower().endswith(('.png', '.jpg', '.jpeg')):
                print('Rostro: ', nameDir + '/' + fileName)
                labels.append(label)
                img_path = os.path.join(personPath, fileName)
                img = cv2.imread(img_path, 0)
                if img is not None:
                    facesData.append(img)
                else:
                    print(f"No se pudo cargar: {img_path}")
        
        label += 1
    
    return facesData, labels

print('Entrenando modelo...')

try:
    faces, labels = obtenerModelo()
    
    # Verificar que tenemos datos
    if len(faces) == 0:
        print("Error: No se encontraron imágenes para entrenar")
        print("Verifica que tengas imágenes en las carpetas dentro de Data/")
        exit()
    
    print(f"Entrenando con {len(faces)} imágenes...")
    
    # Crear y entrenar el reconocedor
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(labels))
    
    # Guardar el modelo
    model_path = 'FacesModel.xml'
    face_recognizer.write(model_path)
    print(f'Modelo guardado como {model_path}')
    print('Entrenamiento completado!')
    
except Exception as e:
    print(f"Error durante el entrenamiento: {e}")
    print("Asegúrate de estar en el entorno virtual activado")