# La Ballena Solitaria - Chatbot for Psychological First Aid Detection and Containment Specialized in Depression Detection

This repository contains the source code for the Backend (RESTful API) of "The Lonely Whale" system, an auxiliary tool for clinical triage and emotional containment.

The system utilizes a hybrid artificial intelligence architecture (non-agentic) that combines deterministic and probabilistic models to assess the risk of depression and anxiety, always prioritizing user safety through a secure-by-design approach.

This backend is designed to be consumed by a frontend client (web application) hosted in an independent repository.

## System Architecture (3 Phases)

1.  **Phase 1 (Static Profiling):** Clinical clustering via unsupervised learning (**K-Means**) based on psychometric standards (**PHQ-9 and GAD-7**).
2.  **Phase 2 (Dynamic NLP Analysis):** Semantic feature extraction using the **RoBERTuito** transformer and emotional classification via a dense neural network (**Multilayer Perceptron / MLP**) balanced with **SMOTE**.
3.  **Phase 3 (Containment and Safety):** Named Entity Recognition (**NER**) rules engine coupled with a Large Language Model (**Llama-3.3-70b via Groq API**) with dynamic context injection. Automated clinical record generation in PDF format.

## Technologies Used

* **API Framework:** FastAPI (Python)
* **Machine Learning:** Scikit-Learn, Imbalanced-learn (SMOTE)
* **Natural Language Processing (NLP):** HuggingFace Transformers, PyTorch
* **LLM / Text Generation:** Groq API (Llama 3)
* **Report Generation:** Matplotlib, FPDF

## Prerequisites

* Python 3.9 or higher.
* A Groq API Key (`GROQ_API_KEY`).

## Installation and Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-user/your-backend-repo.git](https://github.com/your-user/your-backend-repo.git)
    cd your-backend-repo
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On Mac/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure packages such as `fastapi`, `uvicorn`, `transformers`, `torch`, `scikit-learn`, `groq`, `fpdf`, `matplotlib`, and `python-dotenv` are installed).*

4.  **Configure Environment Variables:**
    Create a `.env` file in the project root and add your API key:
    ```env
    GROQ_API_KEY=your_api_key_here
    ```

## Running the Server

The main server is located in the `api_servidor.py` file. To start it in development mode, run:

```bash
uvicorn api_backend:app --reload --host 0.0.0.0 --port 8000

## API Documentation (Endpoints)

FastAPI automatically generates interactive documentation. Once the server is running, you can visit:
* Swagger UI: `http://localhost:8000/docs`
* ReDoc: `http://localhost:8000/redoc`

### Main Endpoints:

* `POST /api/iniciar_sesion`: Receives sociodemographic data and clinical questionnaire scores. Executes Phase 1 (K-Means).
* `POST /api/chat`: Receives user text. Executes Phase 2 (RoBERTuito + MLP) and Phase 3 (NER + LLM). Returns an empathetic containment response.
* `POST /api/terminar_sesion`: Closes the current session, calculates the probabilistic average, and generates the PDF clinical record. Returns the static URL of the document.

## Frontend Connection
This backend is configured with CORS enabled to allow requests from the client (defaulting to http://localhost:4200 for Angular environments). If the frontend is deployed on another domain, the allow_origins list in api_backend.py must be updated.

## Disclaimer 
This software is a research project for the Samsung Innovation Campus course. It is not a medical device and does not substitute for the diagnosis, treatment, or advice of a mental health professional. It is a tool designed exclusively to function as a qualitative triage aid. In case of any life-threatening emergency, it is recommended to contact local health services (e.g., Línea de la Vida in Mexico: 800 911 2000).

---
