import pandas as pd
import numpy as np
from scipy.optimize import linprog
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('data.csv')

# # Preview the data
# print(df.head())

# Select standardized features and target
features = [
    'texture_mean', 'area_mean', 'smoothness_mean', 'concavity_mean',
       'area_se', 'concavity_se', 'fractal_dimension_se', 'smoothness_worst',
       'concavity_worst', 'symmetry_worst'
]

X = data[features].values
# Standardize selected features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Encode target: Malignant (M) = 1, Benign (B) = -1
y = data['diagnosis'].map({'M': 1, 'B': -1}).values

# Add bias term
X_with_bias = np.hstack([X, np.ones((X.shape[0], 1))])  # [x1, ..., x5, b]
n_samples, n_features_plus_bias = X_with_bias.shape
# n_samples = 100  # Limit to 100 samples for the LP
# print(n_samples)

# Create inequality constraints for LP
# Variables: [w1, ..., w5, b, ξ1, ..., ξn]
A_ub = []
b_ub = []

# Constraints: y_i * (w^T x_i + b) + ξ_i ≥ 1 → -y_i * (w^T x_i + b) - ξ_i ≤ -1
for i in range(n_samples):
    xi = X_with_bias[i]
    yi = y[i]
    constraint = [-yi * xij for xij in xi]
    slack = [0] * n_samples
    slack[i] = -1
    row = constraint + slack
    A_ub.append(row)
    b_ub.append(-1)

# Slack variable constraints: ξ_i ≥ 0 → -ξ_i ≤ 0
# for i in range(n_samples):
#     row = [0] * (n_features_plus_bias + n_samples)
#     row[n_features_plus_bias + i] = -1
#     A_ub.append(row)
#     b_ub.append(0)

# Objective: minimize sum of ξ_i
c = [0] * n_features_plus_bias + [1] * n_samples

# Convert to numpy arrays
A_ub = np.array(A_ub)
b_ub = np.array(b_ub)
c = np.array(c)

# Display shapes to confirm setup
print("Shapes of the matrices:")
print("A_ub:", A_ub.shape)
print("b_ub:", b_ub.shape)
print("c:", c.shape)

# Define bounds: weights and bias can be negative, slack variables must be non-negative
bounds = [(None, None)] * n_features_plus_bias + [(0, None)] * n_samples

# Compute the LP
result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

if result.success:
    # Extract the weights and bias from the result
    w = result.x[:n_features_plus_bias - 1]  # Exclude slack variables
    b = result.x[n_features_plus_bias - 1]  # Bias term
    slack = result.x[6:]  # slack variables
else:
    w = b = slack = None

print("Optimal weights:", w)
print("Optimal bias:", b)

# Print and count the number of slack variables that are greater than one
slack_greater_than_one = np.sum(slack >= 1)
print("Number of slack variables greater than one:", slack_greater_than_one)

result.success, result.message, w, b, slack

# Find the objective value (sum of slack variables)
objective_value = result.fun if result.success else None
print("Objective value (sum of slack variables):", objective_value)