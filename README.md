# AI Human Cognitive Model

## 📋 Description

This project was developed with the goal of studying and analyzing how to run machine learning algorithms locally using **scikit-learn**. It serves as a practical learning exercise to understand the fundamentals of building, training, and exporting ML models in a local Python environment.

The project also explores best practices for structuring a Python project at the local level, including the organization of folders, modules, and scripts. This structure is designed to be scalable and maintainable for future machine learning projects.

Additionally, a simple **Linear Regression model** is trained and exported to predict human cognitive scores based on various lifestyle and health factors.

## 🎯 Objectives

- Learn how to execute machine learning algorithms locally with scikit-learn
- Understand Python project structure (modules, packages, and virtual environments)
- Perform exploratory data analysis (EDA) using pandas and plotly
- Train and evaluate a Linear Regression model
- Export trained models for future predictions using joblib

## 📊 Dataset

The dataset (`human_cognitive_performance.csv`) contains **80,000+ records** with the following features:

| Feature | Description |
|---------|-------------|
| `Age` | Age of the individual |
| `Gender` | Gender (Male, Female, Other) |
| `Sleep_Duration` | Hours of sleep per night |
| `Stress_Level` | Stress level (1-10 scale) |
| `Diet_Type` | Type of diet (Vegetarian, Non-Vegetarian, Vegan) |
| `Daily_Screen_Time` | Hours spent on screens daily |
| `Exercise_Frequency` | Exercise frequency (Low, Medium, High) |
| `Caffeine_Intake` | Daily caffeine intake (mg) |
| `Reaction_Time` | Reaction time (ms) |
| `Memory_Test_Score` | Score on memory test |
| `Cognitive_Score` | **Target variable** - Overall cognitive performance score |

## 🛠️ Tech Stack

- **Python 3.11**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **plotly** - Interactive visualizations
- **scikit-learn** - Machine learning library
- **joblib** - Model serialization
- **Jupyter Notebook** - Interactive development environment

## 📁 Project Structure

```
ai_human_cognitive_model/
├── data/
│   └── human_cognitive_performance.csv    # Dataset
├── models/
│   ├── linear_regression_cognitive_score.joblib  # Trained model
│   └── model_info.joblib                  # Model metadata
├── notebooks/
│   └── analysis.ipynb                     # Main analysis notebook
├── scripts/
│   ├── __init__.py
│   └── main.py                            # Python scripts
├── requirements.txt                       # Project dependencies
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.11 installed on your system

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai_human_cognitive_model
   ```

2. **Create a virtual environment**
   ```bash
   py -3.11 -m venv venv
   ```

3. **Activate the virtual environment**
   
   On Windows (bash):
   ```bash
   source venv/Scripts/activate
   ```
   
   On Windows (cmd):
   ```cmd
   venv\Scripts\activate
   ```
   
   On macOS/Linux:
   ```bash
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the Jupyter Notebook**
   ```bash
   jupyter notebook notebooks/analysis.ipynb
   ```

## 📈 Model Performance

The Linear Regression model was trained to predict `Cognitive_Score` based on lifestyle and health factors. Key metrics:

- **R² Score**: Measures how well the model explains variance in the data
- **RMSE**: Root Mean Squared Error - average prediction error
- **MAE**: Mean Absolute Error - average absolute prediction error

## 🔮 Using the Trained Model

To make predictions with new data:

```python
import joblib
import pandas as pd

# Load the model and metadata
model = joblib.load("models/linear_regression_cognitive_score.joblib")
model_info = joblib.load("models/model_info.joblib")

# Prepare new data
new_data = pd.DataFrame({
    'Age': [30],
    'Sleep_Duration': [7.5],
    'Stress_Level': [5],
    'Daily_Screen_Time': [6.0],
    'Caffeine_Intake': [150],
    'Reaction_Time': [350],
    'Memory_Test_Score': [75],
    'Gender': ['Male'],
    'Diet_Type': ['Vegetarian'],
    'Exercise_Frequency': ['High']
})

# Encode categorical variables
new_data_encoded = pd.get_dummies(
    new_data, 
    columns=model_info['categorical_columns'], 
    drop_first=True
)

# Add missing columns
for col in model_info['feature_columns']:
    if col not in new_data_encoded.columns:
        new_data_encoded[col] = False

# Reorder columns
new_data_encoded = new_data_encoded[model_info['feature_columns']]

# Make prediction
prediction = model.predict(new_data_encoded)
print(f"Predicted Cognitive Score: {prediction[0]:.2f}")
```

## 📝 License

This project is for educational purposes.

## 👤 Author

Cezartdev

Developed as a learning project to understand local machine learning workflows with Python and scikit-learn.
