# 🔥 AI-Based Burnout Risk Detection System

A machine learning web app that predicts employee/student burnout risk using a Logistic Regression model, served via a Flask backend with a modern dark UI.

---

## 📁 Project Structure

```
burnout-app/
├── app.py                  # Flask backend
├── model.pkl               # Trained Logistic Regression model
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # Frontend UI
└── README.md
```

---

## 🚀 How to Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/your-username/burnout-risk-detector.git
cd burnout-risk-detector
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your model**

Place your `model.pkl` file in the root directory (same level as `app.py`).

**4. Run the app**
```bash
python app.py
```

**5. Open in browser**
```
http://127.0.0.1:5000
```

---

## 🧠 Model Details

| Property        | Value                  |
|----------------|------------------------|
| Algorithm       | Logistic Regression    |
| Input Features  | Sleep Hours, Stress Level, Workload Rating, Attendance |
| Output          | Burnout Risk: Low / Medium / High |
| Trained With    | scikit-learn            |
| Dataset         | Synthetic (Google Sheets, ~dirty data cleaned in Python) |

---

## 🛠 Tech Stack

- **Backend:** Python, Flask
- **ML:** scikit-learn, NumPy, Pandas
- **Frontend:** HTML, CSS, Vanilla JS
- **Model Serialization:** Pickle

---

## 👩‍💻 Author

Sona — B.Tech CSE, NIST University  
6th Semester ML Lab Project
