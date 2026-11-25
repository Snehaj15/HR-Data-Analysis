import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load HR dataset
df = pd.read_csv("data/HR Employee Dataset.csv")

# Display first 5 rows
print("\n--- FIRST 5 RECORDS ---")
print(df.head())

# Display dataset dimensions
print("\n--- DATASET SHAPE ---")
print(df.shape)

# Display column names
print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

# Display dataset information
print("\n--- DATASET INFORMATION ---")
df.info()

# Check missing values
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# Check duplicate records
print("\n--- DUPLICATE ROWS ---")
print(df.duplicated().sum())

# Check employee attrition counts
print("\n--- ATTRITION COUNTS ---")
print(df["Attrition"].value_counts())

# Calculate attrition rate
attrition_rate = (df["Attrition"].value_counts(normalize=True)["Yes"]) * 100

print("\n--- EMPLOYEE ATTRITION RATE ---")
print(f"Attrition Rate: {attrition_rate:.2f}%")

# Calculate retention rate
retention_rate = 100 - attrition_rate

print(f"Retention Rate: {retention_rate:.2f}%")

# Department-wise attrition analysis
print("\n--- DEPARTMENT-WISE ATTRITION ---")

department_attrition = df.groupby("Department")["Attrition"].value_counts().unstack()

print(department_attrition)

# Calculate attrition percentage by department
department_rate = (
    df[df["Attrition"] == "Yes"]
    .groupby("Department")
    .size()
    / df.groupby("Department").size()
) * 100

print("\n--- ATTRITION RATE BY DEPARTMENT ---")
print(department_rate.round(2))
# Overtime vs Attrition Analysis
print("\n--- OVERTIME VS ATTRITION ---")

overtime_analysis = pd.crosstab(
    df["OverTime"],
    df["Attrition"],
    normalize="index"
) * 100

print(overtime_analysis.round(2))

# Create age groups
df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[18, 25, 35, 45, 55, 65],
    labels=["18-25", "26-35", "36-45", "46-55", "56-65"],
    include_lowest=True
)

# Age group vs attrition
print("\n--- AGE GROUP VS ATTRITION ---")

age_analysis = pd.crosstab(
    df["AgeGroup"],
    df["Attrition"],
    normalize="index"
) * 100

print(age_analysis.round(2))

# Job Satisfaction vs Attrition
print("\n--- JOB SATISFACTION VS ATTRITION ---")

satisfaction_analysis = pd.crosstab(
    df["JobSatisfaction"],
    df["Attrition"],
    normalize="index"
) * 100

print(satisfaction_analysis.round(2))
# Monthly Income vs Attrition
print("\n--- AVERAGE MONTHLY INCOME VS ATTRITION ---")

salary_analysis = df.groupby("Attrition")["MonthlyIncome"].mean()

print(salary_analysis.round(2))

# Department-wise attrition chart

attrition_by_dept = df[df["Attrition"] == "Yes"].groupby("Department").size()

plt.figure(figsize=(8, 5))

sns.barplot(
    x=attrition_by_dept.index,
    y=attrition_by_dept.values
)

plt.title("Employee Attrition by Department")
plt.xlabel("Department")
plt.ylabel("Number of Employees Who Left")
plt.tight_layout()

plt.savefig("output/department_attrition.png")
plt.show()

# Overtime vs Attrition chart

overtime_attrition = df.groupby("OverTime")["Attrition"].apply(
    lambda x: (x == "Yes").mean() * 100
)

plt.figure(figsize=(7, 5))

sns.barplot(
    x=overtime_attrition.index,
    y=overtime_attrition.values
)

plt.title("Attrition Rate by Overtime")
plt.xlabel("Overtime")
plt.ylabel("Attrition Rate (%)")

plt.tight_layout()
plt.savefig("output/overtime_attrition.png")
plt.show()

# Average Monthly Income vs Attrition

salary_data = df.groupby("Attrition")["MonthlyIncome"].mean().reset_index()

plt.figure(figsize=(7, 5))

sns.barplot(
    data=salary_data,
    x="Attrition",
    y="MonthlyIncome"
)

plt.title("Average Monthly Income: Stayed vs Left")
plt.xlabel("Employee Attrition")
plt.ylabel("Average Monthly Income")

plt.tight_layout()
plt.savefig("output/salary_comparison.png")
plt.show()

# STEP 12: Final HR Analytics Dashboard

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Employee Attrition
sns.countplot(data=df, x="Attrition", ax=axes[0, 0])
axes[0, 0].set_title("Employee Attrition")

# 2. Attrition by Department
sns.countplot(
    data=df,
    x="Department",
    hue="Attrition",
    ax=axes[0, 1]
)
axes[0, 1].set_title("Attrition by Department")
axes[0, 1].tick_params(axis="x", rotation=30, labelsize=9)

# 3. Salary Comparison
sns.barplot(
    data=df,
    x="Attrition",
    y="MonthlyIncome",
    ax=axes[1, 0]
)
axes[1, 0].set_title("Average Monthly Income vs Attrition")

# 4. Age Distribution
sns.boxplot(
    data=df,
    x="Attrition",
    y="Age",
    ax=axes[1, 1]
)
axes[1, 1].set_title("Age Distribution vs Attrition")

plt.suptitle("HR Analytics Dashboard", fontsize=20, fontweight="bold")

plt.tight_layout(pad=3)

plt.savefig("output/hr_dashboard.png", dpi=300, bbox_inches="tight")

plt.show()

# Attrition by Age Group
age_data = df.groupby("Age")["Attrition"].apply(
    lambda x: (x == "Yes").sum()
).reset_index(name="Attrition Count")

plt.figure(figsize=(10, 5))
sns.lineplot(data=age_data, x="Age", y="Attrition Count", marker="o")

plt.title("Employee Attrition by Age")
plt.xlabel("Age")
plt.ylabel("Number of Employees Left")

plt.tight_layout()
plt.savefig("output/age_attrition.png")
plt.show()

# Department-wise Attrition
dept_data = df[df["Attrition"] == "Yes"]["Department"].value_counts()

plt.figure(figsize=(7, 7))
plt.pie(
    dept_data,
    labels=dept_data.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Department-wise Employee Attrition")
plt.savefig("output/department_pie.png")
plt.show()

# Job Satisfaction Heatmap
heatmap_data = df.pivot_table(
    index="Department",
    columns="JobSatisfaction",
    values="EmployeeNumber",
    aggfunc="count"
)

plt.figure(figsize=(9, 5))
sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".0f")

plt.title("Department vs Job Satisfaction")
plt.xlabel("Job Satisfaction Level")
plt.ylabel("Department")

plt.tight_layout()
plt.savefig("output/satisfaction_heatmap.png")
plt.show()