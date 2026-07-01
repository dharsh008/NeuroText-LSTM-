# NeuroText LSTM Generator

A project that utilizes a character-level Long Short-Term Memory (LSTM) neural network to generate text, integrated with a premium React frontend.

## Features
- **Character-Level Prediction**: Learns patterns character by character.
- **Customizable Generation**: Control prompt, length, and "creativity" (Top-K).
- **Modern UI**: Sleek, glassmorphism-based design with typewriter effects.
- **PyTorch Backend**: High-performance neural network implementation.

## Getting Started

### 1. Requirements
Ensure you have Python 3.8+ and Node.js 16+ installed.

### 2. Backend Setup
Navigate to the `backend` folder and install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

Train the model (this will take a few moments):
```bash
python train.py
```

Run the API:
```bash
python app.py
```

### 3. Frontend Setup
Navigate to the `frontend` folder and run development server:
```bash
cd frontend
npm install
npm run dev
```

The application will be available at [http://localhost:5173/](http://localhost:5173/).
