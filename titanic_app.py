{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "1dc24fd7-11c3-46b5-b099-9b3c8cc2ad81",
   "metadata": {},
   "outputs": [
    {
     "ename": "FileNotFoundError",
     "evalue": "[Errno 2] No such file or directory: 'logreg_model.pkl'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mFileNotFoundError\u001b[0m                         Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[1], line 7\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mjoblib\u001b[39;00m\n\u001b[0;32m      6\u001b[0m \u001b[38;5;66;03m# Load your trained model\u001b[39;00m\n\u001b[1;32m----> 7\u001b[0m model \u001b[38;5;241m=\u001b[39m joblib\u001b[38;5;241m.\u001b[39mload(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mlogreg_model.pkl\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[0;32m      9\u001b[0m st\u001b[38;5;241m.\u001b[39mtitle(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m🚢 Titanic Survival Prediction App\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[0;32m     10\u001b[0m st\u001b[38;5;241m.\u001b[39mwrite(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mThis app predicts if a passenger survived the Titanic disaster based on input features.\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n",
      "File \u001b[1;32m~\\anacond\\Lib\\site-packages\\joblib\\numpy_pickle.py:650\u001b[0m, in \u001b[0;36mload\u001b[1;34m(filename, mmap_mode)\u001b[0m\n\u001b[0;32m    648\u001b[0m         obj \u001b[38;5;241m=\u001b[39m _unpickle(fobj)\n\u001b[0;32m    649\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m--> 650\u001b[0m     \u001b[38;5;28;01mwith\u001b[39;00m \u001b[38;5;28mopen\u001b[39m(filename, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mrb\u001b[39m\u001b[38;5;124m'\u001b[39m) \u001b[38;5;28;01mas\u001b[39;00m f:\n\u001b[0;32m    651\u001b[0m         \u001b[38;5;28;01mwith\u001b[39;00m _read_fileobject(f, filename, mmap_mode) \u001b[38;5;28;01mas\u001b[39;00m fobj:\n\u001b[0;32m    652\u001b[0m             \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28misinstance\u001b[39m(fobj, \u001b[38;5;28mstr\u001b[39m):\n\u001b[0;32m    653\u001b[0m                 \u001b[38;5;66;03m# if the returned file object is a string, this means we\u001b[39;00m\n\u001b[0;32m    654\u001b[0m                 \u001b[38;5;66;03m# try to load a pickle file generated with an version of\u001b[39;00m\n\u001b[0;32m    655\u001b[0m                 \u001b[38;5;66;03m# Joblib so we load it with joblib compatibility function.\u001b[39;00m\n",
      "\u001b[1;31mFileNotFoundError\u001b[0m: [Errno 2] No such file or directory: 'logreg_model.pkl'"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import joblib\n",
    "\n",
    "# Load your trained model\n",
    "model = joblib.load(\"logreg_model.pkl\")\n",
    "\n",
    "st.title(\"🚢 Titanic Survival Prediction App\")\n",
    "st.write(\"This app predicts if a passenger survived the Titanic disaster based on input features.\")\n",
    "\n",
    "# Input features from user\n",
    "pclass = st.selectbox(\"Passenger Class (Pclass)\", [1, 2, 3])\n",
    "sex = st.selectbox(\"Sex\", ['male', 'female'])\n",
    "age = st.slider(\"Age\", 0, 100, 30)\n",
    "sibsp = st.number_input(\"Number of Siblings/Spouses aboard (SibSp)\", 0, 10, 0)\n",
    "parch = st.number_input(\"Number of Parents/Children aboard (Parch)\", 0, 10, 0)\n",
    "fare = st.slider(\"Fare\", 0.0, 500.0, 50.0)\n",
    "embarked = st.selectbox(\"Port of Embarkation\", ['S', 'C', 'Q'])\n",
    "\n",
    "# Encode categorical inputs\n",
    "sex_encoded = 1 if sex == 'male' else 0\n",
    "embarked_encoded = {'S': 0, 'C': 1, 'Q': 2}[embarked]\n",
    "\n",
    "# Create feature array\n",
    "features = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])\n",
    "\n",
    "# Predict\n",
    "if st.button(\"Predict\"):\n",
    "    prediction = model.predict(features)[0]\n",
    "    probability = model.predict_proba(features)[0][1]\n",
    "\n",
    "    st.write(\"🔍 Prediction:\", \"Survived\" if prediction == 1 else \"Did Not Survive\")\n",
    "    st.write(\"📊 Probability of Survival:\", f\"{probability:.2f}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7001cc1a-c4f6-4a2e-87ca-9ead2a1d5e98",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
