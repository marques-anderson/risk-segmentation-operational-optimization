# Risk-Based Customer Segmentation & Operational Optimization (Python)

This project analyzes how a financial institution could segment customers by risk and allocate investigative resources more efficiently. Using a synthetic customer dataset, I engineered a risk score, segmented customers into low-, medium-, and high-risk groups, evaluated escalation trends and investigation time, and built a basic logistic regression model to identify drivers of escalation.

## Business Problem

Financial institutions and risk teams must decide where to focus limited investigative resources. The goal of this project is to determine whether customers with higher modeled risk also show higher escalation likelihood and greater investigative effort, and whether those patterns can support better prioritization and operational decision-making.

## Imports and Setup

The project uses:
- **pandas** for data tables and analysis
- **NumPy** for numeric operations and synthetic data generation
- **matplotlib** for basic charts


```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
```

## Create the Base Dataset

The synthetic dataset includes core borrower and case-related fields:
- customer demographics and borrowing characteristics
- a risk score built from borrower factors
- investigation time
- escalation outcome


```
n = 500

data = pd.DataFrame({
    'customer_id': range(1, n + 1),
    'age': np.random.randint(21, 70, n),
    'income': np.random.randint(30000, 120000, n),
    'credit_score': np.random.randint(300, 850, n),
    'loan_amount': np.random.randint(1000, 50000, n),
    'loan_duration': np.random.randint(6, 60, n),
    'existing_loans': np.random.randint(0, 5, n),
})

data.head()
```

## Engineer a Risk Score

A simple risk score is created using:
- lower credit score → higher risk
- larger loan amount → higher risk
- more existing loans → higher risk

This is a simplified scoring formula used to create a realistic pattern for analysis.


```
data['risk_score'] = (
    (700 - data['credit_score']) +
    (data['loan_amount'] / 1000) +
    (data['existing_loans'] * 20)
)

data[['credit_score', 'loan_amount', 'existing_loans', 'risk_score']].head()
```

## Add Investigation Time

Investigation time is tied to risk score, with a small random component added so that higher-risk cases usually take longer, but not perfectly every time.


```
data['investigation_time_days'] = (
    data['risk_score'] / 40 + np.random.randint(1, 5, n)
).round(1)

data[['risk_score', 'investigation_time_days']].head()
```

## Convert Risk Score into Escalation Probability

The raw risk score is normalized into a probability range between roughly **10%** and **90%**. This makes higher-risk customers more likely to escalate, while preserving real-world uncertainty.


```
min_score = data['risk_score'].min()
max_score = data['risk_score'].max()

data['escalation_probability'] = 0.1 + 0.8 * (
    (data['risk_score'] - min_score) / (max_score - min_score)
)

data[['risk_score', 'escalation_probability']].head()
```

## Simulate Escalation Outcomes

A random value is generated for each observation and compared with the escalation probability:
- if random value < escalation probability → escalated = 1
- otherwise → escalated = 0

This creates a more realistic outcome variable than a simple hard cutoff.


```
data['random_value'] = np.random.rand(n)

data['escalated'] = (
    data['random_value'] < data['escalation_probability']
).astype(int)

data[['escalation_probability', 'random_value', 'escalated']].head()
```

## Segment Customers by Risk Level

Customers are segmented into three equally sized groups using **quantile-based binning**:
- Low Risk
- Medium Risk
- High Risk

This makes the results easier to interpret from a business perspective.


```
data['segment'] = pd.qcut(
    data['risk_score'],
    3,
    labels=['Low Risk', 'Medium Risk', 'High Risk']
)

data[['risk_score', 'segment']].head()
```

## Basic Structure Check


```
data.info()
```

## Segment Distribution

Because `qcut` was used, each segment should contain roughly one-third of the observations.


```
data['segment'].value_counts()
```

## Escalation Rate by Segment

This shows the average escalation rate (0/1 mean) for each segment.


```
data.groupby('segment', observed=False)['escalated'].mean()
```

## Average Investigation Time by Segment

This shows the average number of investigation days for each segment.


```
data.groupby('segment', observed=False)['investigation_time_days'].mean()
```

## Combined Summary Table

This table brings the core business metrics together:
- escalation rate
- average investigation time


```
summary = data.groupby('segment', observed=False).agg({
    'escalated': 'mean',
    'investigation_time_days': 'mean'
})

summary = summary.rename(columns={
    'escalated': 'Escalation Rate',
    'investigation_time_days': 'Avg Investigation Time'
})

summary
```

## Visual 1: Average Investigation Time by Risk Segment


```
data.groupby('segment', observed=False)['investigation_time_days'].mean().plot(kind='bar')
plt.title("Average Investigation Time by Risk Segment")
plt.xlabel("Risk Segment")
plt.ylabel("Days")
plt.show()
```

## Visual 2: Escalation Rate by Risk Segment


```
data.groupby('segment', observed=False)['escalated'].mean().plot(kind='bar')
plt.title("Escalation Rate by Risk Segment")
plt.xlabel("Risk Segment")
plt.ylabel("Escalation Rate")
plt.show()
```

## Key Findings from Segmentation

- Higher-risk segments show higher escalation rates.
- Investigation time generally increases with risk level.
- Lower-risk customers still consume investigative effort, which suggests a possible opportunity for more efficient prioritization.

## Build a Logistic Regression Model

A basic logistic regression model is used to estimate which variables are most associated with escalation:
- credit score
- loan amount
- existing loans
- loan duration


```
X = data[['credit_score', 'loan_amount', 'existing_loans', 'loan_duration']]
y = data['escalated']
```


```
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```


```
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```


```
y_pred = model.predict(X_test)
```


```
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

## Model Coefficients

The coefficients help show which features increase or decrease escalation likelihood.

- Positive coefficient → higher values tend to increase escalation likelihood
- Negative coefficient → higher values tend to decrease escalation likelihood


```
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
})

coefficients = coefficients.sort_values(by='Coefficient', ascending=False)
coefficients
```


```
coefficients.set_index('Feature')['Coefficient'].plot(kind='bar')
plt.title("Feature Impact on Escalation (Logistic Regression Coefficients)")
plt.xlabel("Feature")
plt.ylabel("Coefficient Value")
plt.axhline(0)
plt.show()
```

## Model Interpretation

Most variables aligned with expectations, but one showed an unexpected relationship due to overlap between features in the dataset. Because multiple variables contribute to overall risk, the model must separate their effects, which can sometimes produce less intuitive results. In a real-world setting, this would warrant further investigation.

## Final Recommendations

- Prioritize higher-risk segments for faster and more focused review.
- Use segmentation to guide case assignment and workload allocation.
- Reduce effort spent on low-risk segments where escalation likelihood is lower.
- Continue refining risk variables and model assumptions to improve prioritization quality.

## Optional Cleanup

The helper columns below were used only to generate synthetic outcomes and can be removed if a cleaner final dataset is preferred:
- `escalation_probability`
- `random_value`


```
# Uncomment the line below if you want to remove helper columns from the final dataset.
# data = data.drop(columns=['escalation_probability', 'random_value'])
```
