## Telco Churn – Complete ML Solution

### Overview

This project implements a comprehensive machine-learning pipeline to predict customer churn for a telecom company. It covers the entire workflow, from preparing data and training models to serving predictions via an API and web interface on AWS.

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
- **CI/CD pipeline:** GitHub Actions builds and pushes Docker images to Docker Hub, with optional ECS service deployment.  
- **Orchestration:** AWS ECS Fargate runs the container serverlessly.  
- **Networking:** ALB on HTTP:80 forwards traffic to a target group (IP targets on HTTP:8000).  
- **Security:** Security groups restrict traffic—ALB allows inbound on 80, ECS tasks accept inbound only from the ALB SG.  
- **Monitoring:** CloudWatch captures container logs and ECS service events.

### Deployment workflow

1. Code is pushed to `main` → GitHub Actions builds the Docker image and uploads it to Docker Hub.  
2. ECS service is updated (manually or via workflow) to launch the new container version.  
3. ALB performs health checks on `/` at port 8000; healthy tasks receive traffic.  
4. Users can access the Gradio UI at `/ui` or send POST requests to `/predict` through the ALB DNS.