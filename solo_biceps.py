import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees
import time

# Configuración de página
st.set_page_config(layout="wide")

# Inicialización de MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# Inicializar contadores en session state
if 'biceps' not in st.session_state:
    st.session_state.update({
        'biceps': {
            'tot': 0,
            'der': 0,
            'izq': 0,
            'abierto_der': False,
            'cerrado_der': False,
            'abierto_izq': False,
            'cerrado_izq': False
        },
        'contando': False,
        'last_update': 0
    })

# Interfaz - Controles arriba
st.title("💪 Contador de Bíceps en Tiempo Real")

# Fila de controles
control_col1, control_col2, control_col3 = st.columns([1, 1, 2])
with control_col1:
    if st.button('Iniciar/Detener Conteo', key='toggle_contando'):
        st.session_state.contando = not st.session_state.contando
with control_col2:
    if st.button('Resetear Contadores', type='primary'):
        st.session_state.biceps = {
            'tot': 0, 'der': 0, 'izq': 0,
            'abierto_der': False, 'cerrado_der': False,
            'abierto_izq': False, 'cerrado_izq': False
        }
with control_col3:
    camara_activa = st.toggle('Activar Cámara', key='toggle_camara')

# Espacio para la cámara
frame_placeholder = st.empty()

# Contadores debajo (se actualizarán en el bucle)
contador_placeholder = st.empty()
status_placeholder = st.empty()

# Función de conteo (igual que antes)
def contar_biceps(frame_rgb, results, width, height):
    if results.pose_landmarks:
        # Hay que intercambiarlos para que cuadren en el espejo que hay que hacer para la camara
        # Derecho (landmarks 12, 14, 16)    ---- Sería el izquierdo
        # Izquierdo (landmarks 11, 13, 15)  ---- Sería el derecho
        biceps_izq = [
            (int(results.pose_landmarks.landmark[i].x * width), 
             int(results.pose_landmarks.landmark[i].y * height))
            for i in [12, 14, 16]
        ]
        
        biceps_der = [
            (int(results.pose_landmarks.landmark[i].x * width), 
             int(results.pose_landmarks.landmark[i].y * height))
            for i in [11, 13, 15]
        ]
        
        def biceps_calcular_angulo(a, b, c):
            biceps_ba = np.array(a) - np.array(b)
            biceps_bc = np.array(c) - np.array(b)
            biceps_coseno_angulo = np.dot(biceps_ba, biceps_bc) / (np.linalg.norm(biceps_ba) * np.linalg.norm(biceps_bc))
            return degrees(acos(biceps_coseno_angulo))
        
        biceps_angulo_der = biceps_calcular_angulo(biceps_der[0], biceps_der[1], biceps_der[2])
        biceps_angulo_izq = biceps_calcular_angulo(biceps_izq[0], biceps_izq[1], biceps_izq[2])
        
        def actualizar_contador(lado, biceps_angulo):
            abierto = f'abierto_{lado}'
            cerrado = f'cerrado_{lado}'
            
            if biceps_angulo >= 150:
                st.session_state.biceps[abierto] = True
            if st.session_state.biceps[abierto] and not st.session_state.biceps[cerrado] and biceps_angulo <= 50:
                st.session_state.biceps[cerrado] = True
            if st.session_state.biceps[abierto] and st.session_state.biceps[cerrado] and biceps_angulo >= 150:
                st.session_state.biceps[lado] += 1
                st.session_state.biceps['tot'] += 1
                st.session_state.biceps[abierto] = False
                st.session_state.biceps[cerrado] = False
                return True
            return False
        
        if st.session_state.contando:
            biceps_rep_der = actualizar_contador('der', biceps_angulo_der)
            biceps_rep_izq = actualizar_contador('izq', biceps_angulo_izq)
            
            if biceps_rep_der or biceps_rep_izq:
                cv2.putText(frame_rgb, "REP COMPLETA!", (width//2-100, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        
        color_der = (0, 255, 0) if st.session_state.contando else (200, 200, 200)
        color_izq = (0, 255, 0) if st.session_state.contando else (200, 200, 200)
        
        for i in range(2):
            cv2.line(frame_rgb, biceps_der[i], biceps_der[i+1], color_der, 3)
            cv2.line(frame_rgb, biceps_izq[i], biceps_izq[i+1], color_izq, 3)
        
        cv2.putText(frame_rgb, f"D: {int(biceps_angulo_der)}°", (biceps_der[1][0], biceps_der[1][1]-15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_rgb, f"I: {int(biceps_angulo_izq)}°", (biceps_izq[1][0], biceps_izq[1][1]-15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame_rgb

# Bucle principal
if camara_activa:
    cap = cv2.VideoCapture(0)
    
    while camara_activa and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Error de captura de cámara")
            break
            
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        frame_rgb = contar_biceps(frame_rgb, results, frame.shape[1], frame.shape[0])
        
        # Mostrar frame
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        # Actualizar contadores
        contador_placeholder.markdown(f"""
        ### 📊 Contadores
        **Derecho:** `{st.session_state.biceps['der']}`  
        **Izquierdo:** `{st.session_state.biceps['izq']}`  
        **Total:** `{st.session_state.biceps['tot']}`
        """)
        
        status_placeholder.success("Conteo ACTIVADO" if st.session_state.contando 
                                 else "Conteo DESACTIVADO")
        
        # Pequeña pausa para reducir carga de CPU
        time.sleep(0.01)
    
    cap.release()
else:
    frame_placeholder.info("Cámara desactivada")
