import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# =====================================================================
# FIRST: Basic Steps
# =====================================================================

# Load dataset
df = pd.read_csv("labor_substitution.csv")

# Show head, shape, columns, dtypes
print("--- Dataset Head ---")
print(df.head())
print("\n--- Dataset Shape ---")
print(df.shape)
print("\n--- Dataset Columns ---")
print(df.columns.tolist())
print("\n--- Dataset Data Types ---")
print(df.dtypes)

# Check and handle missing values 
print("\n--- Missing Values (Before) ---")
print(df.isnull().sum())
df.dropna(inplace=True)
print("\n--- Missing Values (After) ---")
print(df.isnull().sum())

# Remove duplicates
print("\n--- Duplicates (Before) ---")
print(df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("\n--- Duplicates (After) ---")
print(df.duplicated().sum())

# =====================================================================
# SECOND: EDA
# =====================================================================

# Summary statistics
print("\n--- Summary Statistics ---")
print(df.describe())

# Basic understanding of dataset
print("\n--- Dataset Info ---")
df.info()

# =====================================================================
# THIRD: Outlier Detection and Handling
# =====================================================================

# Choose one important numerical column
col = 'Human_Labor_Cost_hr'

# Show boxplot (Before)
plt.figure(figsize=(8, 4))
sns.boxplot(x=df[col], color='lightcoral')
plt.title(f"Boxplot of {col} (Before Outlier Removal)")
plt.show()

# Detect outliers using IQR
Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Print before and after comparison
shape_before = df.shape
df_clean = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
shape_after = df_clean.shape

print("\n--- Outlier Handling Comparison ---")
print(f"Shape Before: {shape_before}")
print(f"Shape After:  {shape_after}")

# Handle outliers properly (Updating the working dataframe)
df = df_clean.copy()

# Show boxplot (After)
plt.figure(figsize=(8, 4))
sns.boxplot(x=df[col], color='lightgreen')
plt.title(f"Boxplot of {col} (After Outlier Removal)")
plt.show()

# =====================================================================
# AFTER THIS: 8 STRONG OBJECTIVES
# =====================================================================

# Objective 1: What is the average Human Labor Cost per Hour for each Industry?
plt.figure(figsize=(10, 5))
avg_cost_industry = df.groupby('Industry')['Human_Labor_Cost_hr'].mean().sort_values()
sns.barplot(x=avg_cost_industry.index, y=avg_cost_industry.values, palette='Blues')
plt.title("Q1: Average Human Labor Cost per Hour by Industry")
plt.xlabel("Industry")
plt.ylabel("Average Cost ($/hr)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Objective 2: How is the Estimated Substitution Year distributed across the roles?
plt.figure(figsize=(8, 5))
sns.histplot(df['Substitution_Year_Est'], bins=10, kde=True, color='purple')
plt.title("Q2: Distribution of Estimated Substitution Year")
plt.xlabel("Estimated Substitution Year")
plt.ylabel("Count of Roles")
plt.tight_layout()
plt.show()

# Objective 3: What is the relationship between Human Labor Cost and Agent Labor Equivalent Cost?
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Human_Labor_Cost_hr', y='Agent_Labor_Equivalent_Cost', data=df, color='orange')
plt.title("Q3: Human Labor Cost vs. Agent Labor Equivalent Cost")
plt.xlabel("Human Labor Cost per Hour")
plt.ylabel("Agent Labor Equivalent Cost")
plt.tight_layout()
plt.show()

# Objective 4: Does the level of Regulatory Moat affect the Automation Risk Index?
plt.figure(figsize=(8, 5))
sns.boxplot(x='Regulatory_Moat', y='Automation_Risk_Index', data=df, order=['Low', 'Med', 'High'], palette='Set2')
plt.title("Q4: Automation Risk Index grouped by Regulatory Moat")
plt.xlabel("Regulatory Moat Level")
plt.ylabel("Automation Risk Index")
plt.tight_layout()
plt.show()

# Objective 5: Which Industry exhibits the highest Average AI Augmentation Factor?
plt.figure(figsize=(10, 5))
avg_ai_aug = df.groupby('Industry')['AI_Augmentation_Factor'].mean().sort_values(ascending=False)
sns.barplot(x=avg_ai_aug.index, y=avg_ai_aug.values, palette='Greens_r')
plt.title("Q5: Average AI Augmentation Factor by Industry")
plt.xlabel("Industry")
plt.ylabel("Average AI Augmentation Factor")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Objective 6: Can we predict Agent Labor Equivalent Cost using Inference Cost 2026? (Linear Regression)
X = df[['Inference_Cost_2026']]
y = df['Agent_Labor_Equivalent_Cost']

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='gray', label='Actual Data')
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')
plt.title("Q6: Predicting Agent Cost based on Inference Cost (2026)")
plt.xlabel("Inference Cost 2026")
plt.ylabel("Agent Labor Equivalent Cost")
plt.legend()
plt.tight_layout()
plt.show()

# Print metrics for the regression 
print(f"Objective 6 - Linear Regression R2 Score: {r2_score(y, y_pred):.4f}")

# Objective 7: How do Tokens per Human Hour vary across different Regulatory Moats?
plt.figure(figsize=(8, 5))
sns.violinplot(x='Regulatory_Moat', y='Tokens_per_Human_Hour', data=df, order=['Low', 'Med', 'High'], palette='mako')
plt.title("Q7: Distribution of Tokens per Human Hour by Regulatory Moat")
plt.xlabel("Regulatory Moat")
plt.ylabel("Tokens per Human Hour")
plt.tight_layout()
plt.show()

# Objective 8: Is there a correlation between Hardware CapEx Sensitivity and Substitution Elasticity?
plt.figure(figsize=(8, 5))
sns.regplot(x='Hardware_CapEx_Sensitivity', y='Substitution_Elasticity', data=df, scatter_kws={'color':'teal'}, line_kws={'color':'black'})
plt.title("Q8: Hardware CapEx Sensitivity vs. Substitution Elasticity")
plt.xlabel("Hardware CapEx Sensitivity")
plt.ylabel("Substitution Elasticity")
plt.tight_layout()
plt.show()
