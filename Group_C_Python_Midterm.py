# Data Handling
import pandas as pd
import numpy as np

#Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# Processing
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Module Building
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load Dataset
df = pd.read_csv("social_media_productivity_6000.csv") 

print(df.head())
# Introduction / Dataset Overview



# Data Loading and Initial Exploration
# Data Inspection
# Data Cleaning
# Data Processing
# Exploratory Data Analysis (EDA)
# Data Visualization
# Machine Learning Model Building
# Model Evaluation
# Conclusion 