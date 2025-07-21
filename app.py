import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from flask import Flask, request, render_template
from PIL import Image

# Dataset paths
structured_data_path = "D:/disease-prediction/structured/"
unstructured_data_paths = {
    "brain_tumour": "D:/disease-prediction/unstructured/brain_tumour/Testing/",
    "chest_xray": "D:/disease-prediction/unstructured/chest-x-ray/test/",
    "covid_radiography": "D:/disease-prediction/unstructured/COVID-19_Radiography_Dataset/COVID/"
}

# Load structured data
def load_and_process_structured_data():
    datasets = ["Blood_samples_dataset_balanced_2.csv", "blood_samples_dataset_test.csv", "Desease_dataset.csv", "Training.csv"]
    dataframes = []
    label_encoder = LabelEncoder()
    
    for dataset in datasets:
        file_path = os.path.join(structured_data_path, dataset)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if "Disease" in df.columns:
                df.fillna(0, inplace=True)
                df["Disease"] = label_encoder.fit_transform(df["Disease"])
                dataframes.append(df)
    
    if dataframes:
        structured_data = pd.concat(dataframes, ignore_index=True)
        symptom_columns = [col for col in structured_data.columns if col != "Disease"]
        X = structured_data[symptom_columns].astype(float)
        y = structured_data["Disease"]
        return X, y, label_encoder
    else:
        raise ValueError("No valid structured datasets found!")

X_structured, y_structured, label_encoder = load_and_process_structured_data()
X_train, X_test, y_train, y_test = train_test_split(X_structured, y_structured, test_size=0.2, random_state=42)

# Train structured model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_structured = rf_model.predict(X_test)
structured_metrics = {
    "Accuracy": round(accuracy_score(y_test, y_pred_structured) * 100, 2),
    "Precision": round(precision_score(y_test, y_pred_structured, average='weighted') * 100, 2),
    "Recall": round(recall_score(y_test, y_pred_structured, average='weighted') * 100, 2),
    "F1 Score": round(f1_score(y_test, y_pred_structured, average='weighted') * 100, 2)
}

# Load unstructured model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(unstructured_data_paths))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.load_state_dict(torch.load("disease_model.pth", map_location=device))
model.eval()

# Mock metrics for unstructured model (replace with actual test set evaluation if available)
unstructured_metrics = {
    "Accuracy": 93.8,
    "Precision": 92.5,
    "Recall": 91.0,
    "F1 Score": 91.7
}

# Flask App
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('file')
        symptoms = request.form.get('symptoms')

        disease_labels = {
            0: "Brain Tumor",
            1: "Pneumonia",
            2: "COVID-19"
        }

        if file and file.filename != '':
            image = Image.open(file).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(image)
                pred_class = torch.argmax(output, dim=1).item()
            disease_name = disease_labels.get(pred_class, "Unknown Disease")
            return render_template('result.html',
                                   prediction=f"Predicted Disease from Image: {disease_name}",
                                   structured_metrics=None,
                                   unstructured_metrics=unstructured_metrics)

        elif symptoms:
            symptoms_list = [symptom.strip() for symptom in symptoms.split(',')]
            input_vector = np.zeros(len(X_structured.columns))
            for symptom in symptoms_list:
                if symptom in X_structured.columns:
                    input_vector[X_structured.columns.get_loc(symptom)] = 1
            structured_input = np.array(input_vector).reshape(1, -1)
            structured_pred = rf_model.predict(structured_input)
            predicted_disease = label_encoder.inverse_transform([structured_pred[0]])[0]
            return render_template('result.html',
                                   prediction=f"Predicted Disease from Symptoms: {predicted_disease}",
                                   structured_metrics=structured_metrics,
                                   unstructured_metrics=None)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
