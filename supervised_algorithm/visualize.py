import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from linear_regression import F, T, X

# Create output folder if it doesn't exist
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# Select first feature: MedInc (Median Income, standardized)
x_feature = X[:, 0]

# Create a side-by-side subplot comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Left plot: Ground truth target values
axes[0].scatter(x_feature, T, color='red', alpha=0.25, s=8)
axes[0].set_title("Ground Truth: MedInc vs Actual Price")
axes[0].set_xlabel("Median Income (Standardized)")
axes[0].set_ylabel("Price ($100k)")
axes[0].grid(True, linestyle="--", alpha=0.5)

# Right plot: Model predictions
axes[1].scatter(x_feature, F, color='green', alpha=0.25, s=8)
axes[1].set_title("Model Output: MedInc vs Predicted Price")
axes[1].set_xlabel("Median Income (Standardized)")
axes[1].grid(True, linestyle="--", alpha=0.5)

# Adjust layout and export
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "feature_comparison_split.png"), dpi=200)
plt.close()

print("Plot saved to plots/feature_comparison_split.png")