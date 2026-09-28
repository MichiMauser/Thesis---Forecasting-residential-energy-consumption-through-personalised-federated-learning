# Predicția consumului energetic rezidențial prin învăţare federată cu personalizare

**Author:** Eduard-Mihai Moga
**Coordinators:** Prof. dr. ing. Rodica Potolea, S.l. dr. ing. Raluca Portase
**Institution:** Technical University of Cluj-Napoca (UTCN), Computer Science Department

---

## 📖 Overview

This repository contains the implementation for my Bachelor's thesis, which focuses on designing, implementing, and evaluating a privacy-preserving machine learning system for short-term residential energy forecasting at the individual smart meter level[cite: 1, 2]. 

The system predicts energy consumption for the next 30-minute interval directly on edge devices, ensuring that highly sensitive personal consumption data never leaves the household[cite: 1, 2]. To overcome the strong statistical heterogeneity (non-IID data) inherent to individual households, the project implements a federated learning architecture with a dedicated personalization phase (Per-FedAvg)[cite: 1, 2]. This approach achieves predictive accuracy nearly identical to locally specialized models[cite: 1, 2].

## ⚙️ Core Architecture & Features

### 1. Machine Learning Modeling
*   **Residual LSTM Network:** A lightweight Long Short-Term Memory network (~78,000 parameters) that predicts the deviation ($\Delta$) from the last known value, ensuring the model never performs worse than a simple persistence baseline on stable consumption periods[cite: 1, 2].
*   **Robust Optimization:** To handle highly skewed, long-tailed consumption distributions, the target variable undergoes a `log1p` transformation, and the network is optimized using a Huber loss function ($\delta=0.5$)[cite: 1, 2].
*   **Feature Engineering:** Utilizes 23 features including cyclic temporal encoding, historical lags (1, 2, 48, 336), 48-hour rolling statistics, explicit holiday indicators, and meteorological data[cite: 1, 2].

### 2. Federated Learning (FL) Framework
*   **Distributed Deployment:** Built on the Flower framework using gRPC, where the server and each client operate as fully independent processes[cite: 1, 2].
*   **Federation Algorithms:** Evaluates and compares FedAvg, FedProx, and Per-FedAvg (fine-tuning)[cite: 1, 2].
*   **Personalization (Per-FedAvg):** Uses the federated global model as an initialization point for local fine-tuning, successfully mitigating client drift[cite: 1, 2].
*   **Diversity-Based Client Selection:** Ensures the global model learns from a representative population by selecting clients using a greedy farthest-point sampling algorithm based on five statistical features[cite: 1, 2].

### 3. Live Monitoring & Orchestration System
*   **FastAPI Backend:** Orchestrates the training process and parses telemetry data recorded in a `JSONL` run log[cite: 1, 2].
*   **Vue 3 + Vite Frontend:** A Single-Page Application (SPA) dashboard that tracks the federated training process in real-time using Server-Sent Events (SSE)[cite: 1, 2].

## 📊 Dataset & Results

*   **Dataset:** Evaluated using the public **London Smart Meters** dataset (Low Carbon London project), containing semi-hourly records for 5,566 residential units[cite: 1, 2].
*   **Key Result:** The Per-FedAvg personalized approach achieves a mean $R^2$ score of 0.530 across highly diverse clients, nearly matching the theoretical local upper bound (0.539) and significantly outperforming standard FedAvg (0.496) and FedProx (0.457)[cite: 1, 2].

---

## 💻 System Requirements

*   **Hardware:** Minimum 4-core CPU and 8 GB RAM. A dedicated GPU is supported but not strictly required. Note: Running more than ~10 clients simultaneously on a single machine will saturate the CPU and extend round durations.
*   **Software:** Python 3.10+. Node.js 18+ is only required if you plan to rebuild the frontend UI.

## 🚀 Installation

1. **Install Dependencies**  
   From the root directory, install all required Python packages:
   ```bash
   pip install -r requirements.txt


   Prepare the DatasetPlace the London Smart Meters dataset files in the root directory. Required files:   Semi-hourly consumption blocks (block*.csv)   Household metadata (informations_households.csv)   Hourly weather data (weather_hourly_darksky.csv)   

   Usage GuideInteraction with the system is handled entirely through the web interface; no code or config file editing is required. The UI is organized into five main sections: Control, Live Board, Clients List, Client Detail, and Comparison.   1. Start the ServerLaunch the application in standard mode to serve both the FL orchestrator and the precompiled frontend on a single port:   Bashpython -m uvicorn app.backend.main:app --port 8000
Navigate to http://localhost:8000 in your web browser.   
2. Configure and Start a Session (Control Page)Select Clients: Search for households by ID or socio-economic group and add between 2 and 20 units to your selection, then save.   Start Daemons: Click the start button to launch the independent daemon processes (one for each selected household).   Set Parameters: Choose the federation algorithm (FedAvg, FedProx, or Per-FedAvg). Set the number of communication rounds, local epochs, the proximal term $\mu$ (if using FedProx), and customization epochs (if using Per-FedAvg).   Trigger Run: Click to begin the federated training session. You will be automatically redirected to the live monitoring page.   
3. Track the Ongoing Federation (Live Board)Communication Graph: Visualizes the connection between the server and clients. Node colors indicate performance, and edge thickness reflects the client's drift from the global model.   Convergence Metrics: A live graph tracks the evolution of the $R^2$ coefficient and Mean Absolute Error across rounds.   Results: Upon completion, the gallery of test-set predictions for each client will load automatically.   
4. Client and Comparative AnalysisClient List & Details: View the status and primary metrics for each client. Click into a specific client to view its Exploratory Data Analysis (EDA), training curves, and individual predictions.   Comparison Page: Compare the performance of different training methods (Local, FedAvg, FedProx, Per-FedAvg) side-by-side using interactive bar charts, and view overlaid consumption distributions to understand client heterogeneity.   Teardown: When finished, use the stop button on the control page to terminate the daemon processes and free system resources.   Development Mode (For UI Modifications)To modify the Vue 3 frontend, run the backend and the Vite development server concurrently:
Start Backend: python -m uvicorn app.backend.main:app --port 8000 
Start Frontend: cd app/frontend && npm install && npm run dev
Access the development UI at http://localhost:5173 
Once finished, compile the  UI using npm run build