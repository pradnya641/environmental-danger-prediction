# Environmental Danger Prediction System

An AI-powered environmental monitoring and danger prediction system designed to identify potential hazards from visual scenes, assess their associated risks, and maintain incident records through an interactive monitoring dashboard.

The system integrates **Gemini Vision, LangGraph, FastAPI, React, and MySQL** to create an intelligent hazard detection and incident management pipeline.

## Overview

Environmental hazards in workplaces and public spaces can lead to accidents, injuries, and property damage. Early identification and risk assessment can help improve environmental safety and support timely intervention.

This project aims to automate the process of environmental hazard identification using AI-powered visual analysis. It analyzes images and live camera frames, identifies potential hazards, evaluates their risk levels, and stores the resulting incidents for monitoring and review.

## Key Features

- **AI-Based Hazard Detection:** Uses Gemini Vision to analyze images and identify potential environmental hazards.
- **Live Camera Monitoring:** Captures webcam frames through the frontend for continuous visual analysis.
- **Risk Assessment:** Evaluates detected hazards and assigns corresponding risk levels.
- **Agent-Based Processing:** Uses LangGraph to organize the analysis pipeline.
- **Incident Management:** Stores detected incidents and associated hazard information in MySQL.
- **Interactive Dashboard:** Displays incident statistics, risk distributions, and monitoring information.
- **Incident History:** Supports viewing, searching, and filtering recorded incidents.
- **Incident Status Management:** Allows incident statuses to be updated through the backend API.

## System Architecture

The system follows a modular architecture consisting of a frontend, backend, AI processing pipeline, and database.

```text
        Image / Live Webcam Frame
                   |
                   v
          React Frontend
                   |
                   v
             FastAPI Backend
                   |
                   v
             LangGraph Pipeline
                   |
                   v
          Gemini Vision Analysis
                   |
                   v
             Risk Assessment
                   |
                   v
          Incident Processing
                   |
                   v
              MySQL Database
                   |
                   v
          Monitoring Dashboard
```

## Technology Stack

| Component | Technologies |
|---|---|
| Frontend | React, TypeScript, Vite |
| Backend | Python, FastAPI |
| AI Vision | Google Gemini Vision |
| Workflow Orchestration | LangGraph |
| Database | MySQL |
| ORM | SQLAlchemy |
| API Communication | REST API |

## Project Structure

```text
environmental-danger-prediction/
│
├── agents/
│   ├── detection_agent.py
│   ├── gemini_scene_agent.py
│   └── risk_agent.py
│
├── frontend/
│   ├── public/
│   └── src/
│       ├── App.tsx
│       ├── App.css
│       ├── LiveMonitoring.tsx
│       ├── LiveMonitoring.css
│       └── main.tsx
│
├── pipeline/
│   ├── graph.py
│   └── video_graph.py
│
├── tests/
│
├── utils/
│   └── video_processor.py
│
├── app.py
├── database.py
├── models.py
├── run_video.py
└── .gitignore
```

## Application Workflow

1. **Input Acquisition:** The system receives an uploaded image or captures a frame from a live webcam.
2. **Scene Analysis:** Gemini Vision analyzes the visual input to identify potential environmental hazards.
3. **Risk Evaluation:** The risk assessment agent evaluates the analysis and determines the associated risk level.
4. **Pipeline Processing:** LangGraph coordinates the processing workflow.
5. **Incident Storage
