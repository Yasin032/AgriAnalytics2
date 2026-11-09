# Agri Analytics

A Streamlit-based agricultural analytics dashboard for crop insights, recommendations, and yield forecasting.

## Local run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## GitHub and Streamlit deployment

1. Create a repository on GitHub.
2. Push this project to GitHub:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

3. In Streamlit Cloud, click New app and select the repository.
4. Set the app main file to `app.py`.
5. Deploy.

## Requirements

The project uses:
- streamlit
- pandas
- numpy
- plotly
- scikit-learn
