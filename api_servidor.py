import joblib
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from groq import Groq  
from datetime import datetime
import time
import matplotlib.pyplot as plt
from fpdf import FPDF
import os
from dotenv import load_dotenv

# Importaciones para el servidor web
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# ==========================================
# 0. CONFIGURACIÓN DE LA API EN LA NUBE
# ==========================================
load_dotenv()
api_key_groq = os.getenv("GROQ_API_KEY") 
# Si no usas .env, ponla directo: api_key_groq = "TU_API_KEY_AQUÍ"
client = Groq(api_key=api_key_groq)

# ==========================================
# 1. CARGA DE CEREBROS (FASES 1 Y 2)
# ==========================================
print("Cargando cerebro del sistema (Esto puede tomar unos segundos)...")
tokenizer = AutoTokenizer.from_pretrained("pysentimiento/robertuito-base-uncased")
robertuito = AutoModel.from_pretrained("pysentimiento/robertuito-base-uncased")
mlp_afecto = joblib.load('modelos/modelo_MPL_afecto.pkl')

scaler_f1 = joblib.load('modelos/scaler_clustering.pkl')
kmeans_f1 = joblib.load('modelos/modelo_clustering_v1.pkl')

sumas_centros = kmeans_f1.cluster_centers_.sum(axis=1)
ID_VERDE = sumas_centros.argmin()
ID_ROJO = sumas_centros.argmax()

# ==========================================
# 2. DICCIONARIO Y FUNCIONES DE TRIAGE
# ==========================================
DICCIONARIO_CLINICO = {
    'PHQ9_Anhedonia': ['apatía', 'desmotivad', 'aburrid', 'interés', 'placer', 'hueva', 'sin ganas', 'flojera', 'nada me importa'],
    
    'PHQ9_Estado_Animo': ['deprimid', 'sin esperanza', 'miserable', 'triste', 'infeliz', 'agüitad', 'bajón', 'oscuridad', 'llorar', 'sol', 'vací'],
    
    'PHQ9_Somático': ['insomnio', 'sueño', 'dormir', 'exhaust', 'fatiga', 'energía', 'cansad', 'pesad', 'débil', 'agotamient', 'apetito', 'peso', 'comer', 'hambre', 'lent', 'no rindo', 'desvelad'],
    
    'PHQ9_Cognitivo_Autoestima': ['fracaso', 'inútil', 'culpa', 'aislad', 'estorbo', 'carga', 'vergüenza', 'decepcionad', 'concentración', 'atención', 'decisiones', 'enfocarme', 'olvidadiz', 'tont', 'brut', 'no sirvo'],
    
    'GAD7_Fisico': ['corazón', 'pecho', 'respirar', 'aire', 'tembland', 'temblor', 'sudor', 'náusea', 'mareo', 'entumecid', 'hormigueo', 'estómago', 'ahogand', 'tensión', 'taquicardia'],
    
    'GAD7_Mental': ['pánico', 'ataque', 'ansios', 'ansiedad', 'preocupación', 'miedo', 'asustad', 'pensamientos', 'sobrepensar', 'obsesiv', 'irracional', 'control', 'loc', 'angustia'],
    
    'Burnout_Situacional': ['abrumad', 'presión', 'entrega', 'carga', 'quemad', 'burnout', 'batalland', 'colapso', 'irritable', 'cabeza', 'exigente', 'deudas', 'dinero', 'lana', 'escuela', 'universidad', 'uni', 'prepa', 'trabajo', 'chamba', 'responsabilidades', 'nervios', 'demasiado', 'gente', 'estrés', 'estresad', 'harto', 'ya no puedo'],
    
    'Alerta_Suicida': ['matarme', 'matar', 'suicidio', 'suicidarme', 'suicidar', 'terminar con todo', 'terminar con mi vida','termino con mi vida','quitarme la vida', 'morir', 'muerte', 'adiós', 'carta', 'pastillas', 'sobredosis', 'puente', 'colgarme', 'mejor muert', 'sin salida', 'harto de la vida', 'rendirme', 'acabar con todo', 'no vale la pena', 'desaparecer', 'no despertar', 'ya no quiero vivir', 'dormir para siempre', 'cortarme', 'hacerme daño', 'dejar de existir,desvivirme','dedsvivo','desvivir'],
    
    'Afecto_Positivo': ['hoy', 'personas', 'pensar', 'amig', 'familia', 'vida', 'bien', 'chido', 'feliz', 'ok', 'mejor', 'tranquil', 'calma', 'esperanza', 'motivad', 'descansé']
}

def motor_ner_definitivo(texto):
    categorias_detectadas = []
    texto_limpio = texto.lower()
    for categoria, palabras_clave in DICCIONARIO_CLINICO.items():
        if any(palabra in texto_limpio for palabra in palabras_clave):
            categorias_detectadas.append(categoria)
    return categorias_detectadas

def puente_clinico_sin_sesgo(datos_usuario, scaler_cargado, kmeans_cargado, id_verde, id_rojo):
    cols_clinicas = ['Growing_Stress', 'Coping_Struggles', 'Work_Interest', 'Mood_Swings', 'Changes_Habits', 'Social_Weakness', 'Mental_Health_History', 'Quarantine_Frustrations', 'Weight_Change']
    reg = {
        'Growing_Stress': datos_usuario.get('estres', 0), 'Coping_Struggles': datos_usuario.get('afrontamiento', 0),
        'Work_Interest': datos_usuario.get('interes', 0), 'Mood_Swings': datos_usuario.get('humor', 0),
        'Changes_Habits': datos_usuario.get('habitos', 0), 'Social_Weakness': datos_usuario.get('debilidad_social', 0),
        'Mental_Health_History': datos_usuario.get('historial', 0), 'Quarantine_Frustrations': datos_usuario.get('irritabilidad', 0),
        'Weight_Change': datos_usuario.get('peso', 0)
    }
    
    df_in = pd.DataFrame([reg], columns=cols_clinicas)
    df_in_scaled = scaler_cargado.transform(df_in)
    cluster_ia = kmeans_cargado.predict(df_in_scaled)[0]
    
    sintomas_base = [reg['Growing_Stress'], reg['Coping_Struggles'], reg['Work_Interest'], reg['Mood_Swings'], reg['Changes_Habits'], reg['Social_Weakness']]
    puntuacion_real = sum(sintomas_base)
    
    print("\n" + "- "*15)
    print("   [PROCESANDO TRIAGE MATEMÁTICO - FASE 1]")
    print("- "*15)
    print(f" -> 1. Extrayendo variables clínicas: {list(reg.values())}")
    print(f" -> 2. Escalando datos (StandardScaler)...")
    print(f" -> 3. K-Means (IA) asignó el cluster: {cluster_ia}")
    print(f" -> 4. Calculando Carga Sintomatológica (Suma): {puntuacion_real}/12 puntos")
    
    if puntuacion_real <= 2:
        if reg['Mental_Health_History'] == 2 or cluster_ia == id_rojo: 
            return "RIESGO MODERADO"
        print(" -> 5. Dictamen: RIESGO BAJO (Puntuación dentro de 0-2).")
        return "RIESGO BAJO"
    
    elif 3 <= puntuacion_real <= 5:
        if cluster_ia == id_rojo or reg['Mental_Health_History'] == 2: 
            print(" -> 5. Regla de Blindaje: Sube a ALTO por clúster crítico o historial.")
            return "RIESGO ALTO"
        print(" -> 5. Dictamen: RIESGO MODERADO (Puntuación dentro de 3-5).")
        return "RIESGO MODERADO"
    else: 
        print(" -> 5. Dictamen: RIESGO ALTO (Puntuación >= 6, umbral clínico superado).")
        return "RIESGO ALTO"

def procesar_triage_unificado(datos_cuestionario, mensaje_chat):
    riesgo_base = puente_clinico_sin_sesgo(datos_cuestionario, scaler_f1, kmeans_f1, ID_VERDE, ID_ROJO)
    
    inputs = tokenizer(mensaje_chat, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad(): outputs = robertuito(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].numpy()

    probabilidades = mlp_afecto.predict_proba(embedding)[0]
    indices_ordenados = np.argsort(probabilidades)[::-1] 
    
    idx_principal = indices_ordenados[0]
    idx_secundario = indices_ordenados[1] 
    confianza = probabilidades[idx_principal]
    
    mapeo_afecto = {0: 'Normal', 1: 'Stress', 2: 'Anxiety', 3: 'Depression', 4: 'Suicidal'}
    estado_actual = mapeo_afecto[idx_principal]   
    
    sintomas_detectados = motor_ner_definitivo(mensaje_chat)
    alerta_critica = False
    
    if "Alerta_Suicida" in sintomas_detectados:
        alerta_critica = True
        estado_actual = "Suicidal"
        confianza = 1.0
        probabilidades = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        
    elif estado_actual == "Suicidal":
        if confianza < 0.998:
            estado_actual = mapeo_afecto[idx_secundario] 
            alerta_critica = False
        else:
            alerta_critica = True

    return {
        "perfil_riesgo_f1": riesgo_base, 
        "afecto_detectado_f2": estado_actual, 
        "confianza_f2": confianza, 
        "sintomas_ner": sintomas_detectados, 
        "alerta_critica": alerta_critica,
        "probabilidades_crudas": probabilidades
    }

def interactuar_con_ia(historial_mensajes):
    intentos = 0
    max_intentos = 3
    
    while intentos < max_intentos:
        try:
            chat_completion = client.chat.completions.create(
                messages=historial_mensajes,
                model="llama-3.3-70b-versatile",
                temperature=0.7,
                max_tokens=300
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            error_str = str(e).lower()
            intentos += 1
            if "rate limit" in error_str or "429" in error_str or "too many requests" in error_str:
                print(f"  [!] Límite de velocidad de la API. Esperando 6 segundos (Intento {intentos}/{max_intentos})...")
                time.sleep(6)
            else:
                return f"[ERROR_API_TOKENS] Detalles: {e}"
    return f"[ERROR_API_TOKENS] Límite de reintentos superado: {e}"

# ==========================================
# 3. GENERACIÓN DEL PDF
# ==========================================
def generar_reporte_y_grafica(historial_probabilidades, datos_usuario, riesgo_global, nombre_usuario="Paciente"):
    print("\n📊 Generando gráfica y reporte clínico detallado...")
    
    carpeta_destino = "pdf"
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)
    
    if historial_probabilidades:
        promedios = np.mean(historial_probabilidades, axis=0)
    else:
        promedios = [0, 0, 0, 0, 0]

    # Gráfica PNG
    etiquetas = ['Normal', 'Estrés', 'Ansiedad', 'Depresión', 'Riesgo Suicida']
    colores = ['#4CAF50', '#FFC107', '#FF9800', '#9C27B0', '#F44336']
    plt.figure(figsize=(8, 5))
    barras = plt.bar(etiquetas, promedios * 100, color=colores)
    plt.title(f'Perfil Emocional Promedio de la Sesión (MLP)', fontsize=14)
    plt.ylabel('Porcentaje de Confianza (%)')
    plt.ylim(0, 100)
    
    for barra in barras:
        yval = barra.get_height()
        posicion_y = yval - 5 if yval > 10 else yval + 1
        color_texto = 'white' if yval > 10 else 'black'
        plt.text(barra.get_x() + barra.get_width()/2, posicion_y, f'{yval:.1f}%', ha='center', va='center', color=color_texto, fontweight='bold')

    ruta_grafica = os.path.join(carpeta_destino, 'grafica_emociones.png')
    plt.tight_layout()
    plt.savefig(ruta_grafica)
    plt.close()

    # Documento PDF
    pdf = FPDF(orientation='P', unit='mm', format='Letter')
    pdf.add_page()
    
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, 'REPORTE PSICOLÓGICO AUTOMATIZADO', new_x="LMARGIN", new_y="NEXT", align='C')
    
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_text_color(0, 51, 153) 
    pdf.cell(0, 5, 'Documento de apoyo para valoración profesional', new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 7, f'Identificador/Nombre: {nombre_usuario}', new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 5, f'Fecha de evaluación: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, f'Perfil: {datos_usuario.get("age_cat")}, {datos_usuario.get("ocupacion")}', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, 'I. Hallazgos matemáticos (cuestionario de seguimiento)', new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('Helvetica', '', 9)
    
    map_resp = {0: "No / Nunca", 1: "A veces", 2: "Casi siempre"}
    preguntas_pdf = [
        ("Estrés/Relajación", datos_usuario.get('estres')),
        ("Anhedonia/Interés", datos_usuario.get('interes')),
        ("Energía/Herramientas", datos_usuario.get('afrontamiento')),
        ("Estado de ánimo/Tristeza", datos_usuario.get('humor')),
        ("Aislamiento social", datos_usuario.get('debilidad_social')),
        ("Irritabilidad/Frustración", datos_usuario.get('irritabilidad')),
        ("Hábitos/Sueño/Peso", datos_usuario.get('habitos')),
        ("Tiempo de aislamiento", datos_usuario.get('aislamiento')),
        ("Historial clínico previo", datos_usuario.get('historial'))
    ]

    for preg, valor in preguntas_pdf:
        texto_resp = map_resp.get(valor, "Sí, con tratamiento" if valor == 2 else "Sin dato")
        pdf.cell(80, 6, f'- {preg}:', border=0)
        pdf.set_font('Helvetica', 'B', 9)
        pdf.cell(0, 6, f'{texto_resp}', new_x="LMARGIN", new_y="NEXT", border=0)
        pdf.set_font('Helvetica', '', 9)

    pdf.ln(5)
    
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, 'II. Análisis predominante en la conversacion', new_x="LMARGIN", new_y="NEXT")
    pdf.image(ruta_grafica, x=35, w=140)
    pdf.ln(5)

    pdf.set_x(10)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 10, 'III. Conclusión y observaciones técnicas', new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.ln(4)

    p_normal = promedios[0] * 100
    p_estres = promedios[1] * 100
    p_ansiedad = promedios[2] * 100
    p_depresion = promedios[3] * 100
    p_suicida = promedios[4] * 100

    if p_suicida > 5:
        status_emergencia = "ALERTA CRÍTICA (requiere protocolo de contención inmediato)"
    else:
        status_emergencia = "Negativo (sin riesgo vital inminente detectado en el discurso)"

    emociones_porcentajes = [
        ('Tranquilidad/Normalidad', p_normal), ('Estrés', p_estres), 
        ('Ansiedad', p_ansiedad), ('Depresión', p_depresion), ('Riesgo Suicida', p_suicida)
    ]
    emociones_ordenadas = sorted(emociones_porcentajes, key=lambda x: x[1], reverse=True)
    top1 = emociones_ordenadas[0]
    top2 = emociones_ordenadas[1]

    # APARTADO A
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 8, 'A. Resultados Específicos del Paciente', new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('Helvetica', '', 9)

    resultados_paciente = (
        f"Dictamen general (cuestionario de seguimiento): El paciente es clasificado con {riesgo_global}.\n\n"
        f"Estatus de riesgo autolítico (conversación): {status_emergencia}.\n\n"
        f"Para esta sesión, el modelo detectó que las emociones predominantes expresadas en el discurso fueron "
        f"{top1[0]} ({top1[1]:.1f}%) y {top2[0]} ({top2[1]:.1f}%)."
    )
    pdf.multi_cell(0, 5, resultados_paciente, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    hallazgo_normal = f"- Normal ({p_normal:.1f}%): " + ("El paciente mostró un discurso estable con presencia de afecto positivo." if p_normal > 40 else "Los momentos de tranquilidad o emociones agradables fueron poco frecuentes.")
    hallazgo_estres = f"- Estrés ({p_estres:.1f}%): " + ("Se detectó sobrecarga mental, frustración o quejas ambientales significativas en la charla." if p_estres > 25 else "No se detectó sobrecarga situacional aguda en la conversación.")
    hallazgo_ansiedad = f"- Ansiedad ({p_ansiedad:.1f}%): " + ("Se observó sobrepensamiento, miedo al futuro o mención de síntomas físicos." if p_ansiedad > 25 else "El discurso no evidenció ideación ansiosa desbordante.")
    hallazgo_depresion = f"- Depresión ({p_depresion:.1f}%): " + ("ALERTA: Se identificó desesperanza, apatía o tristeza profunda durante la sesión." if p_depresion > 25 else "Sin marcadores fuertes de discurso depresivo.")
    hallazgo_suicida = f"- Riesgo Suicida ({p_suicida:.1f}%): " + ("CRÍTICO: Se detectaron menciones de deseos de muerte o no tener salida." if p_suicida > 5 else "Ausencia de marcadores inminentes de riesgo vital.")

    for hallazgo in [hallazgo_normal, hallazgo_estres, hallazgo_ansiedad, hallazgo_depresion, hallazgo_suicida]:
        pdf.multi_cell(0, 5, hallazgo, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # APARTADO B
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 8, 'B. Observaciones Técnicas y Simbología Clínica', new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('Helvetica', '', 9)

    obs_tecnicas = (
        "Es imperativo destacar que este chatbot y su motor NLP fueron desarrollados ÚNICA Y EXCLUSIVAMENTE con el objetivo de "
        "detectar sintomatología y patrones de DEPRESIÓN. Las demás métricas (ansiedad, estrés, normal) se incluyen únicamente "
        "como covariables para entender el contexto clínico del paciente.\n\n"
        "Simbología clínica de la gráfica y extracción de entidades:"
    )
    pdf.multi_cell(0, 5, obs_tecnicas, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    
    simb_normal = "- Normal: Discurso estable y congruente. El paciente denota presencia de afecto positivo (expresiones de tranquilidad, motivación, esperanza), y capacidad de resolución de problemas. No se aprecian alteraciones graves en el estado de ánimo."
    simb_estres = "- Estrés: Carga alostática notable. Detecta sobrecarga mental por responsabilidades cotidianas o indicios de Burnout (ej. quejas sobre falta de tiempo, presiones académicas/laborales)."
    simb_ansiedad = "- Ansiedad: Métrica basada en los criterios del GAD-7. Detecta sobrepensamiento significativo, preocupación excesiva e irracional hacia el futuro, o menciones de síntomas somáticos (taquicardia, falta de aire)."
    simb_depresion = "- Depresión: MÉTRICA OBJETIVO. Basado estrictamente en los criterios del PHQ-9. Identifica lenguaje vinculado a anhedonia (pérdida de interés), desesperanza profunda, culpa excesiva o tristeza clínica (ej. 'no tengo ganas de nada', 'siento que soy un fracaso')."
    simb_suicida = "- Riesgo Suicida: Ideación autolítica. Menciones explícitas o implícitas de deseos de muerte, autolesión, sensación de no tener salida o despedida (ej. 'ya no quiero estar aquí', 'mejor desaparecer')."

    for simb in [simb_normal, simb_estres, simb_ansiedad, simb_depresion, simb_suicida]:
        pdf.multi_cell(0, 5, simb, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

    pdf.ln(3)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 4, "Este documento es el resultado de un procesamiento algorítmico de lenguaje natural (NLP) y Machine Learning. No constituye un diagnóstico médico definitivo. Sirve exclusivamente como herramienta de tamizaje (Triage) auxiliar para el profesional de la salud.", new_x="LMARGIN", new_y="NEXT")

    nombre_archivo = f"Reporte_Tecnico_{nombre_usuario.replace(' ', '_')}.pdf"
    ruta_pdf_final = os.path.join(carpeta_destino, nombre_archivo)
    pdf.output(ruta_pdf_final)
    print(f"[OK] Archivos guardados en carpeta '{carpeta_destino}'")

# ==========================================
# 4. CONFIGURACIÓN DEL SERVIDOR FASTAPI
# ==========================================
app = FastAPI(title="API Ballena Azul - Tesis")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if not os.path.exists("pdf"):
    os.makedirs("pdf")
app.mount("/pdf", StaticFiles(directory="pdf"), name="pdf")

# ==========================================
# 5. MEMORIA Y MODELOS DE DATOS WEB
# ==========================================
sesion_actual = {
    "datos_usuario": {},
    "historial_chat": [],
    "historial_probabilidades": [],
    "riesgo_global_f1": "",
    "turnos": 0,
    "alerta_activa": False
}

class RespuestasCuestionario(BaseModel):
    nombre: str
    edad: str
    ocupacion: str
    respuestas_clinicas: list[int] 

class MensajeUsuario(BaseModel):
    texto: str



# ... (Importaciones y carga de modelos se mantienen igual hasta la sección de Endpoints)

# ==========================================
# 5. MEMORIA (Estado persistente del servidor)
# ==========================================
sesion_actual = {
    "datos_usuario": {},
    "historial_chat": [], # Aquí se guardará el historial para Groq
    "historial_probabilidades": [],
    "riesgo_global_f1": "",
    "turnos": 0,
    "alerta_activa": False
}

# ==========================================
# 6. ENDPOINTS
# ==========================================

@app.post("/api/iniciar_sesion")
async def iniciar_sesion(datos: RespuestasCuestionario):
    global sesion_actual
    # Reiniciamos todo para una nueva charla
    sesion_actual["historial_chat"] = []
    sesion_actual["historial_probabilidades"] = []
    sesion_actual["turnos"] = 0
    sesion_actual["alerta_activa"] = False
    
    r = datos.respuestas_clinicas
    sesion_actual["datos_usuario"] = {
        'age_cat': datos.edad, 'ocupacion': datos.ocupacion, 'nombre': datos.nombre, 
        'estres': r[0], 'interes': r[1], 'afrontamiento': r[2], 'humor': r[3],
        'debilidad_social': r[4], 'irritabilidad': r[5], 'habitos': r[6], 
        'peso': r[6], 'aislamiento': r[7], 'historial': r[8]
    }
    return {"mensaje": "Sesión iniciada correctamente."}

@app.post("/api/chat")
async def procesar_chat(mensaje: MensajeUsuario):
    global sesion_actual
    
    # --- VERIFICAR EL LÍMITE CLÍNICO (Equivalente al 'break' de tu while) ---
    if sesion_actual["turnos"] >= 12:
        print("\n[SISTEMA]: El protocolo de primeros auxilios psicológicos ha concluido.")
        return {
            "respuesta": "🤖 IA: Nuestro tiempo de sesión ha terminado por hoy. Ha sido un paso muy valiente hablar de esto. Por favor cuídate mucho, generaré tu reporte clínico.",
            "emocion_detectada": "Normal",
            "alerta": sesion_actual["alerta_activa"],
            "finalizar": True
        }
        
    sesion_actual["turnos"] += 1
    
    texto_usuario = mensaje.texto
    datos_usuario = sesion_actual["datos_usuario"]
    historial = sesion_actual["historial_chat"]
    
    # Analizamos el mensaje uniendo la Fase 1 y Fase 2
    reporte = procesar_triage_unificado(datos_usuario, texto_usuario)
    sesion_actual["riesgo_global_f1"] = reporte['perfil_riesgo_f1']
    sesion_actual["historial_probabilidades"].append(reporte['probabilidades_crudas'])
    
    # MODO DEBUG (Se imprimirá en la consola de tu servidor)
    info_emocional = f"MLP: {reporte['afecto_detectado_f2']} (Seguridad: {reporte['confianza_f2']:.1%}) | NER: {reporte['sintomas_ner']}"
    print(f"  [DEBUG FASE 2] -> {info_emocional}")
    
    # ==========================================
    # VÍA A: SAFETY GATE (CRISIS)
    # ==========================================
    if reporte['alerta_critica']:
        print("\n🚨 [ALERTA INTERNA]: Riesgo Crítico/Suicida Detectado. Activando protocolo de contención continua.")
        sesion_actual["alerta_activa"] = True
        
        prompt_crisis = """
        ROL Y PROPÓSITO:
        Eres un especialista en intervención en crisis. El usuario acaba de expresar intenciones o pensamientos suicidas (RIESGO CRÍTICO).
        
        REGLAS ESTRICTAS DE COMPORTAMIENTO:
        1. ÚNICO OBJETIVO: Tu objetivo absoluto es mantener a la persona platicando contigo para anclarla al presente, validar su dolor profundo sin juzgar, y persuadirla suavemente de que busque ayuda.
        2. NÚMERO DE EMERGENCIA: En tu respuesta, de forma empática, pídele que llame a la Línea de la Vida al 800 911 2000. Diles que hay profesionales listos para ayudarles.
        3. LONGITUD Y CIERRE: Responde en máximo 4 oraciones. TERMINA SIEMPRE con una pregunta abierta que invite al usuario a seguir desahogándose contigo (ej. "¿Qué es lo que más te duele en este momento?", "¿Quieres contarme más sobre eso? Estoy aquí").
        4. NO TERMINES LA CONVERSACIÓN.
        """
        
        if len(historial) == 0:
            historial.append({"role": "system", "content": prompt_crisis})
        else:
            # Sobrescribimos el prompt normal con el de crisis
            historial[0] = {"role": "system", "content": prompt_crisis}
            
    # ==========================================
    # VÍA B: LLAMADA A GROQ (CHAT NORMAL)
    # ==========================================
    else:
        instruccion = ""
        if "BAJO" in reporte['perfil_riesgo_f1']: instruccion = "Sé amigable y cálido. Enfócate en mantener su bienestar."
        elif "MODERADO" in reporte['perfil_riesgo_f1']: instruccion = "Sé cauteloso, explora su nivel de estrés y valida sus emociones."
        elif "ALTO" in reporte['perfil_riesgo_f1']: instruccion = "Ofrece contención prioritaria, sé muy delicado y sugiere fuertemente buscar apoyo psicológico."

        # Traducimos la emoción del MLP al español para que la IA la integre mejor
        diccionario_traduccion = {'Stress': 'estrés', 'Anxiety': 'ansiedad', 'Depression': 'tristeza o depresión', 'Normal': 'tranquilidad'}
        emocion_traducida = diccionario_traduccion.get(reporte['afecto_detectado_f2'], reporte['afecto_detectado_f2'])

        # Psicoeducación basada en la confianza (>75%)
        if reporte['confianza_f2'] > 0.75 and reporte['afecto_detectado_f2'] != 'Normal':
            regla_psicoeducacion = f"2. PSICOEDUCACIÓN: Si el usuario expresa malestar, menciona sutilmente que podría estar relacionado con síntomas de {emocion_traducida}. NUNCA des un diagnóstico definitivo. SIN EMBARGO, si el usuario está notoriamente feliz, motivado o tranquilo, IGNORA la métrica clínica, no menciones síntomas y acompáñalo en su alegría y logro."
        else:
            regla_psicoeducacion = "2. PSICOEDUCACIÓN: Escucha activamente. Debido a la ambigüedad del mensaje, NO menciones diagnósticos ni síntomas específicos en este turno, solo acompaña sus emociones (sean positivas o negativas) de forma muy empática."
        
        # Formatear lista de síntomas para el prompt
        sintomas_str = ', '.join(reporte['sintomas_ner']) if reporte['sintomas_ner'] else 'Ninguno'
        
        prompt_sistema = f"""
        ROL Y PROPÓSITO:
        Eres un asistente de primeros auxilios psicológicos y contención emocional. NO eres un médico, psiquiatra ni psicólogo clínico. Tu objetivo es escuchar, validar emociones y brindar psicoeducación.

        CONTEXTO DEL PACIENTE:
        - Edad: {datos_usuario.get('age_cat')}
        - Ocupación: {datos_usuario.get('ocupacion')}
        - Pronombres preferidos: {datos_usuario.get('nombre')}
        - Riesgo Base (Fase 1): {reporte['perfil_riesgo_f1']}
        - Emoción actual detectada (Fase 2): {emocion_traducida}
        - Síntomas clínicos detectados en su mensaje: {sintomas_str}

        REGLAS ESTRICTAS DE COMPORTAMIENTO:
        1. TONO: {instruccion} Sé conversacional y empático.
        {regla_psicoeducacion}
        3. DISCLAIMER: Incluye una frase natural indicando que eres una IA y no sustituyes atención médica, solo es necesario que se EXPRESE UNA SOLA VEZ ,concentrate en escuchar.
        4. LONGITUD: Responde en máximo 4 oraciones. 
        4. REGLA DE EMERGENCIA (PLAN B): Si el usuario menciona querer morir, hacer daño o despedirse, DEBES responder textualmente recomendando la Línea de la Vida en México (800 911 2000) y preguntarle qué siente en este momento.
        """
        
        if sesion_actual["turnos"] >= 10:
            prompt_sistema += "\nINSTRUCCIÓN CRÍTICA: La sesión está por terminar. Empieza a concluir la charla de forma muy cálida, dale un consejo final y NO le hagas preguntas que alarguen la plática."
        else:
            prompt_sistema += "\nTermina siempre con una pregunta suave para invitarlo a desahogarse."
        
        if len(historial) == 0:
            historial.append({"role": "system", "content": prompt_sistema})
        else:
            historial[0] = {"role": "system", "content": prompt_sistema}
            
    # Agregamos el mensaje del usuario al historial
    historial.append({"role": "user", "content": texto_usuario})
    
    print("  ... Consultando a la IA ...") 
    respuesta_ia = interactuar_con_ia(historial)
    
    # Manejo de fallos en la API (Caída elegante)
    if "[ERROR_API_TOKENS]" in respuesta_ia:
        print("\n❌ [ALERTA DEL SISTEMA]: Fallo de conexión o límite en Groq.")
        respuesta_ia = "🤖 IA: Disculpa, mi sistema ha alcanzado el límite de carga y debemos pausar nuestra charla. Ha sido un paso muy valiente buscar ayuda. Por favor, guarda tu reporte clínico y considera buscar apoyo con un profesional."
        sesion_actual["alerta_activa"] = True
        
    print(f"🤖 IA: {respuesta_ia}\n")
    
    # Agregamos la respuesta de la IA al historial
    historial.append({"role": "assistant", "content": respuesta_ia})
    
    return {
        "respuesta": respuesta_ia, 
        "emocion_detectada": reporte['afecto_detectado_f2'],
        "alerta": sesion_actual["alerta_activa"],
        "finalizar": sesion_actual["turnos"] >= 12
    }

@app.post("/api/terminar_sesion")
async def terminar_sesion():
    global sesion_actual
    nombre_usuario = sesion_actual["datos_usuario"].get("nombre", "Paciente")
    
    generar_reporte_y_grafica(
        sesion_actual["historial_probabilidades"], 
        sesion_actual["datos_usuario"], 
        sesion_actual["riesgo_global_f1"], 
        nombre_usuario
    )
    
    nombre_archivo = f"Reporte_Tecnico_{nombre_usuario.replace(' ', '_')}.pdf"
    url_publica = f"http://localhost:8000/pdf/{nombre_archivo}"
    return {"url_pdf": url_publica}

# ==========================================
# 7. ARRANQUE DEL SERVIDOR
# ==========================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)