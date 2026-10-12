# 💊 MedAlert AI
### AI-Powered Medicine Inventory & Demand Forecasting System

MedAlert AI is a Python-based inventory management web application that uses Machine Learning to estimate medicine demand and recommend stock replenishment.

## 🚀 Features
- Medicine inventory management
- Medicine expiry monitoring
- Low-stock alerts
- ML-based medicine demand prediction
- Intelligent reorder quantity recommendations
- Purchase-order drafts with user approval
- SQLite database integration
- Interactive Streamlit dashboard

## 🧠 Machine Learning
- Algorithm: Random Forest Regressor
- Input features: Date, weekday, month, and available stock
- Target: Daily medicine sales
- Evaluation metric: Mean Absolute Error (MAE)
- Dataset: Synthetic sample medicine sales data

## 🛠️ Tech Stack
- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- SQLite
- Plotly

## ▶️ Run Locally

Install dependencies:

```bash
pip install -r requirements.txt