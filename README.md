# Disease-prediction-using-Generative-ai-and-MLFlow
🧠 Disease Prediction System using Generative AI & Machine Learning
Empowering Healthcare with AI
Predict diseases using structured data, unstructured symptom text, and medical imaging — integrated with MLflow for seamless experiment tracking.

📌 Project Overview
In this project, we explore the integration of Generative AI and Machine Learning for disease prediction, leveraging MLflow to ensure reproducibility, traceability, and systematic experimentation.

The healthcare industry increasingly relies on AI-powered diagnostic systems to enable early and accurate disease detection. This project presents a hybrid approach that combines both structured medical data (e.g., symptoms, test results) and unstructured data (e.g., textual input, medical images).

🚀 Key Features
🔍 Structured Data Prediction using a Random Forest classifier.

🧬 Unstructured Data Prediction through transformer-based Generative AI models.

🧠 Image-based Detection using deep learning (e.g., ResNet50 for brain tumor/chest X-ray).

📊 MLflow Integration to track experiments, hyperparameters, model versions, and metrics.

🌐 Flask Web App to provide real-time predictions through an intuitive interface.

⚙️ Technologies & Tools
Category	Tools & Frameworks
Languages	Python
Machine Learning	Scikit-learn, RandomForestClassifier
Deep Learning	PyTorch, CNN, ResNet50
Generative AI	Transformer-based models for text-based inference
Experiment Tracking	MLflow
Web Development	Flask, HTML5, CSS3 (Jinja2 templates)
Visualization	Matplotlib, Seaborn
Environment	virtualenv, VS Code

🗂 Project Structure
graphql
Copy
Edit
disease-prediction/
├── app.py                 # Flask application
├── models/                # Saved ML/DL models
├── structured/            # CSV datasets (structured features)
│   ├── Blood_samples_dataset_balanced.csv
│   ├── Disease dataset.csv
│   └── Testing.csv
├── unstructured/          # Medical image datasets
│   ├── brain_tumour/
│   ├── chest-x-ray/
│   └── COVID-19_Radiography_Dataset/
├── templates/             # Web UI templates (index.html, result.html)
├── uploads/               # Uploaded test images
├── disease_model.pth      # Trained PyTorch model
├── mlruns/                # MLflow experiment tracking folder
└── venv/                  # Virtual environment
📈 Experiment Tracking with MLflow
We implemented MLflow to:

Track and log model parameters, metrics, and artifacts

Compare model versions and evaluate performance

Maintain reproducibility and auditability for healthcare deployments

Example: Compare Random Forest vs Transformer-based models in real time using MLflow UI.

🧪 Model Summary
Model Type	Input Type	Description
Random Forest	Structured Data	Symptom and test-based prediction
Transformer (GenAI)	Unstructured Text	Predicts diseases from patient-provided symptoms
CNN (ResNet50)	Medical Images	Classifies brain tumors and chest anomalies

🏁 How to Run
Clone the Repository

bash
Copy
Edit
git clone https://github.com/yourusername/disease-prediction.git
cd disease-prediction
Set Up Environment

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate (Windows)
pip install -r requirements.txt
Start the App

bash
Copy
Edit
python app.py
Access MLflow (Optional)

bash
Copy
Edit
mlflow ui
# Visit http://localhost:5000 to explore experiments
📊 Results & Insights
Improved accuracy through combined data modalities

Generative AI enhances prediction from symptom descriptions

MLflow ensures robust tracking, versioning, and comparison of models

🏥 Impact
This project demonstrates how AI, particularly Generative AI and Machine Learning, can augment healthcare diagnostics. It showcases the potential of combining structured and unstructured data with a modern AI pipeline that includes experiment management, reproducibility, and user-facing deployment.

📄 License
This project is open-source under the MIT License.
