# Exoplanet AI 🌌

A machine learning-powered web application to predict the habitability of exoplanets based on their physical and stellar characteristics.

## 🚀 Features

-   **Habitability Prediction**: Uses a Random Forest classifier to predict if an exoplanet is potentially habitable.
-   **Interactive Dashboard**: Visualize feature importance, habitability distribution, and parameter correlations using Plotly.
-   **Excel Export**: Download a report of the top candidate habitable planets.
-   **REST API**: Exposes a `/predict` endpoint for programmatic access.

## 🛠️ Tech Stack

-   **Python 3.10+**
-   **Flask**: Web framework
-   **Scikit-learn**: Machine learning
-   **Pandas & NumPy**: Data processing
-   **Plotly**: Interactive visualizations
-   **Bootstrap 5**: Responsive UI

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/exoplanet-ai.git
    cd exoplanet-ai
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    # Windows
    .\.venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🏃 Usage

1.  **Run the application:**
    ```bash
    python app.py
    ```
    The browser will automatically open to the dashboard.

2.  **Access the Dashboard:**
    Go to `http://127.0.0.1:5000/dashboard` to view insights.

3.  **Predict:**
    Go to `http://127.0.0.1:5000/` to use the prediction form.

## 📂 Project Structure

```
├── app.py                  # Main Flask application
├── model_training.py       # Script to train and save the model
├── rf_model.pkl           # Trained Random Forest model
├── exoplanet_data_processed.csv # Processed dataset
├── templates/
│   ├── base.html           # Base HTML template
│   ├── index.html          # Prediction form
│   └── dashboard.html      # Visualization dashboard
├── requirements.txt        # Python dependencies
└── Procfile                # Heroku deployment configuration
```

## ☁️ Deployment (Heroku/Render)

1.  Ensure `Procfile` and `requirements.txt` are present.
2.  Push to GitHub.
3.  Connect your repository to Heroku or Render.
4.  Deploy!

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## 📄 License

MIT License