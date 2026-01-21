# Exoplanet AI: Hunting for Habitable Worlds 🪐

Hi there! 👋 Welcome to **Exoplanet AI**.

Have you ever looked up at the night sky and wondered, *"Is there another Earth out there?"* I certainly have. That curiosity led me to build this project—a machine learning application designed to help us identify planets beyond our solar system that might just be capable of supporting life.

## 🤔 What is this project?

In simple terms, this is a smart tool that analyzes data from thousands of discovered exoplanets. Using a "Random Forest" machine learning model (think of it as a committee of decision-making trees), it looks at key characteristics like a planet's size, temperature, and the type of star it orbits to predict whether it could be habitable.

I built this to bridge the gap between complex astronomical data and easy-to-understand insights.

## ✨ What can you do here?

-   **Predict Habitability**: Got some data on a new planet? Enter it into the simple form on the homepage, and the AI will tell you if it's a potential candidate for life.
-   **Explore the Data**: Check out the **Dashboard** to see beautiful, interactive charts. You can see which features matter most for life (like planet radius or temperature) and visualize the "Habitability Score" of known planets.
-   **Download Findings**: Want to dive deeper? You can export a list of the top 100 most promising habitable planets directly to an Excel file.

## 🛠️ How it's built

For the tech-savvy, here's what's under the hood:
-   **Brain**: Python & Scikit-learn (Random Forest Classifier).
-   **Web Interface**: Flask (a lightweight web framework).
-   **Visuals**: Plotly (for those interactive charts).
-   **Data Processing**: Pandas & NumPy.

## 🚀 Getting Started

Want to run this on your own machine? Here is the step-by-step:

1.  **Clone the code**:
    ```bash
    git clone https://github.com/GirishsarmaMitta22/ExoplanetAi-Girish-Sarma.git
    cd ExoplanetAi-Girish-Sarma
    ```

2.  **Set up the environment**:
    It's best to use a virtual environment so you don't mess with your system's Python.
    ```bash
    python -m venv .venv
    .\.venv\Scripts\activate  # On Windows
    ```

3.  **Install the necessary tools**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Launch!**:
    ```bash
    python app.py
    ```
    Your browser should automatically open up to the dashboard. If not, just click the link shown in your terminal.

## ☁️ Deployment

This app is ready to fly! It includes a `Procfile` and `requirements.txt` so you can easily deploy it to platforms like Heroku or Render.

## 👋 About the Author

This project was built by **Girish Sarma**. I'm passionate about space, AI, and building things that matter. If you have questions or want to contribute, feel free to reach out or open an issue on GitHub!

---
*"Somewhere, something incredible is waiting to be known." - Carl Sagan*