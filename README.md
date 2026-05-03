# 🌫️ Air Pollution Predictor

A deep learning-based project to predict air pollution levels using time-series data. This project focuses on comparing **LSTM** and **CNN-LSTM** models to analyze whether more complex architectures improve prediction accuracy.

---

## 📖 Overview

Air pollution has severe impacts on human health and the environment. Accurate prediction of pollution levels can help in early warnings and policy decisions.

This project:
- Uses historical air quality data  
- Builds deep learning models for prediction  
- Compares performance between models  
- Draws insights on model complexity vs accuracy  

---

## 🚀 Features

- Data preprocessing and cleaning  
- Exploratory Data Analysis (EDA)  
- Implementation of LSTM and CNN-LSTM models  
- Performance evaluation using metrics like RMSE and MAE  
- Visualization of predictions vs actual values  

---

## 🧠 Models

### LSTM (Long Short-Term Memory)
- Designed for sequential/time-series data  
- Captures temporal dependencies effectively  

### CNN-LSTM Hybrid
- CNN extracts feature patterns  
- LSTM handles temporal relationships  
---

## 📊 Dataset

The dataset contains historical air pollution data such as PM2.5, PM10, NO₂, CO, and SO₂, along with time-based features like date and time for time-series forecasting.

---

Paper on Comparision of various models : https://archive.org/details/mark-khan-research-paper

## ⚙️ Installation

```bash
git clone https://github.com/markkhannnn/Air-Pollution-Predictor.git
cd Air-Pollution-Predictor
pip install -r requirements.txt
