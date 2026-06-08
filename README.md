# ☀️ New Delhi Weather Predictor

An end-to-end Machine Learning pipeline predicting tomorrow's temperature in New Delhi based on historical weather APIs and lag features. 

### Tech Stack
* **Machine Learning:** Scikit-Learn (Random Forest Regressor), Pandas, Numpy
* **Data Source:** Open-Meteo Historical Archive API
* **Backend:** FastAPI, Uvicorn
* **Frontend:** Streamlit

### How to Run Locally
1. Clone this repository.
2. Create a virtual environment and install the requirements:
   `pip install -r requirements.txt`
3. (Optional) Retrain the model on the latest data:
   `python train_model.py`
4. Start the backend API:
   `uvicorn api:app --reload`
5. Open a new terminal and start the UI:
   `streamlit run app.py`