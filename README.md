# La Ballena Solitaria - Chatbot para la Detección y Contención de Primeros Auxilios Psicológicos Especializado para la Detección de Depresión

Este repositorio contiene el código fuente del **Backend (API RESTful)** para el sistema "La Ballena Solitaria", una herramienta auxiliar de triage clínico y contención emocional. 

El sistema utiliza una **arquitectura híbrida de inteligencia artificial** (no agentica) que combina modelos deterministas y probabilísticos para evaluar el riesgo de depresión y ansiedad, priorizando siempre la seguridad del usuario mediante un diseño seguro.

Este backend está diseñado para ser consumido por un cliente frontend (aplicación web) alojado en un repositorio independiente.

## Arquitectura del Sistema (3 Fases)

1. **Fase 1 (Perfilado estático):** Agrupamiento clínico mediante aprendizaje no supervisado (**K-Means**) basado en estándares psicométricos (PHQ-9 y GAD-7).
2. **Fase 2 (Análisis dinámico NLP):** Extracción de características semánticas con el transformer **RoBERTuito** y clasificación emocional mediante una red neuronal densa (**Perceptrón Multicapa / MLP**) balanceada con SMOTE.
3. **Fase 3 (Contención y seguridad):** Motor de reglas de reconocimiento de entidades nombradas (NER) , acoplado a un modelo de lenguaje grande (**Llama-3.3-70b** vía Groq API) con inyección de contexto dinámico. Generación automatizada de expedientes en formato PDF.

## Tecnologías Utilizadas

* **Framework API:** [FastAPI](https://fastapi.tiangolo.com/) (Python)
* **Machine Learning:** Scikit-Learn, Imbalanced-learn (SMOTE)
* **Procesamiento de Lenguaje Natural (NLP):** HuggingFace Transformers, PyTorch
* **LLM / Generación de Texto:** Groq API (Llama 3)
* **Generación de Reportes:** Matplotlib, FPDF

## Requisitos Previos

* Python 3.9 o superior.
* Una clave de API de Groq (`GROQ_API_KEY`).

## Instalación y Configuración

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/tu-usuario/tu-repo-backend.git
   cd tu-repo-backend
   ```

2. **Crear y activar un entorno virtual:**
   ```bash
   python -m venv venv
   # En Windows:
   venv\Scripts\activate
   # En Mac/Linux:
   source venv/bin/activate
   ```

3. **Instalar las dependencias:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Asegúrate de tener instalados paquetes como `fastapi`, `uvicorn`, `transformers`, `torch`, `scikit-learn`, `groq`, `fpdf`, `matplotlib`, `python-dotenv`).*

4. **Configurar Variables de Entorno:**
   Crea un archivo `.env` en la raíz del proyecto y agrega tu clave de API:
   ```env
   GROQ_API_KEY=tu_clave_api_aqui
   ```

## Ejecución del Servidor

El servidor principal se encuentra en el archivo `api_backend.py`. Para levantarlo en modo desarrollo, ejecuta:

```bash
uvicorn api_backend:app --reload --host 0.0.0.0 --port 8000
```
El servidor estará disponible en `http://localhost:8000`.

## Documentación de la API (Endpoints)

FastAPI genera documentación interactiva automáticamente. Una vez que el servidor esté corriendo, puedes visitar:
* **Swagger UI:** `http://localhost:8000/docs`
* **ReDoc:** `http://localhost:8000/redoc`

### Endpoints Principales:

* `POST /api/iniciar_sesion`: Recibe los datos sociodemográficos y los puntajes del cuestionario clínico. Ejecuta la Fase 1 (K-Means).
* `POST /api/chat`: Recibe el texto del usuario. Ejecuta la Fase 2 (RoBERTuito + MLP) y la Fase 3 (NER + LLM). Devuelve la respuesta empática de contención.
* `POST /api/terminar_sesion`: Clausura la sesión actual, calcula el promedio probabilístico y genera el archivo PDF con el expediente clínico. Devuelve la URL estática del documento.

## Conexión con el Frontend

Este backend está configurado con **CORS** habilitado para permitir peticiones desde el cliente (por defecto `http://localhost:4200` para entornos Angular). Si el frontend se despliega en otro dominio, es necesario actualizar la lista `allow_origins` en `api_backend.py`.

## Aviso 

Este software es un proyecto de investigación para el curso de Samsung Innovation Campus. **No es un dispositivo médico ni sustituye el diagnóstico, tratamiento o consejo de un profesional de la salud mental.** Es una herramienta diseñada exclusivamente para funcionar como un auxiliar de *triage* cualitativo. Ante cualquier emergencia vital, se recomienda contactar a los servicios de salud locales (ej. Línea de la Vida en México: 800 911 2000).

---
