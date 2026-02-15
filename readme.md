## Telco Churn – Complete ML Solution by Lars Mogensen

### Overview

This project implements a comprehensive machine-learning pipeline to predict customer churn for a telecom company. It covers the entire workflow, from data preparation and model training to serving predictions via an API and web interface. The application is containerized for easy deployment on any server or cloud environment.

(Full raw dataset and complete MLflow logs are not included due to size/privacy; a small demo dataset is provided for reproducibility.)

### Why it matters

- **Proactive retention:** Identifies customers at risk of leaving so teams can intervene early.  
- **Accessible ML:** The model is available through a REST API and an easy-to-use web UI, eliminating the need for notebooks.  
- **Consistent delivery:** Containerized deployment and CI/CD pipelines ensure updates are reproducible and reliable.  
- **Experiment tracking:** MLflow records metrics, runs, and artifacts to guarantee reproducibility and auditability.

### Components

- **Data & Modeling:** Feature engineering plus an XGBoost classifier; all experiments logged in MLflow.  
- **Model management:** Metrics, runs, and serialized models tracked under a dedicated MLflow experiment.  
- **Prediction API:** FastAPI service exposing `/predict` (POST) and a root `/` endpoint for health checks.  
- **Web interface:** Gradio UI mounted at `/ui` for easy, shareable testing.  
- **Containerization:** Docker image with `uvicorn` entrypoint (`src.app.main:app`) listening on port 8000.  
- **CI/CD pipeline:** GitHub Actions builds and optionally pushes Docker images to Docker Hub.  
- **Deployment-ready:** The container can be run locally or deployed on any server or cloud platform supporting Docker.  
- **Monitoring:** Logs are captured for debugging and performance tracking.

### Deployment workflow

1. Code is pushed to `main` → GitHub Actions builds the Docker image and optionally uploads it to a container registry.  
2. The Docker container can be run locally or on a server, exposing FastAPI endpoints and the Gradio UI.  
3. Users can access the Gradio UI at `/ui` or send POST requests to `/predict` through the API.
