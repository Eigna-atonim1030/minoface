# 🔍 Sistema de Reconocimiento Facial con ESP32-CAM y Arduino

Proyecto desarrollado en el marco de las Experiencias Académicas de la Facultad de Ingeniería (EAFI 1P-2025) - XIV Edición.

---

## 📄 Introducción

Este proyecto tiene como objetivo facilitar el **control de acceso mediante reconocimiento facial en tiempo real**, combinando tecnologías de **Internet de las Cosas (IoT)** e **Inteligencia Artificial (IA)**. 

La solución captura imágenes con una **ESP32-CAM**, las procesa mediante un script en Python y muestra los resultados en una **pantalla TFT** conectada a un **Arduino Uno**, brindando una solución segura, eficiente y accesible.

---

## 🎯 Objetivos del Proyecto

### 🎯 General
Desarrollar un sistema de control de acceso basado en reconocimiento facial en tiempo real, integrando una **ESP32-CAM** con procesamiento en Python.

### ✅ Específicos
- Capturar imágenes desde la ESP32-CAM.
- Detectar rostros autorizados usando algoritmos de IA.
- Mostrar el resultado en una pantalla TFT conectada a un Arduino Uno.
- Registrar eventos de reconocimiento facial.

---

## 💡 Arquitectura del Sistema

El sistema se estructura en una arquitectura de **4 capas**:

### 🧠 1. Capa de Percepción (Hardware)
- **Componente**: ESP32-CAM SBC OV2640-MODULO WIFI ESP32  
- **Función**: Adquisición de imágenes y video

### 🌐 2. Capa de Red
- **Protocolo**: Servidor HTTP (MJPEG)
- **Conectividad**: Wi-Fi con IP asignada por DHCP

### 🧪 3. Capa de Servicio
- **Tecnologías**: `face_recognition`, Deep Learning, comunicación serial
- **Funciones**: Procesamiento de imágenes, verificación y modelado

### 🖥️ 4. Capa de Aplicación
- **Componentes**: Arduino Uno, pantalla TFT
- **Función**: Visualización en tiempo real (verde = acceso permitido, rojo = denegado)

---

## 🔄 Diagrama de Flujo del Sistema

```text
1. Inicio del sistema
2. Captura de video con ESP32-CAM
3. Detección de rostro con OpenCV
   ↳ No hay rostro → regresar a captura
   ↳ Hay rostro → recorte y envío
4. Reconocimiento facial con Deep Learning
5. Validación:
   ↳ Alta confianza → acceso permitido
   ↳ Baja confianza → acceso denegado
6. Visualización en pantalla TFT
7. Registro del evento
8. Repetición del proceso (bucle continuo)
