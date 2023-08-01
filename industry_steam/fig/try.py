
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Generate data
np.random.seed(1)
dist1 = np.random.normal(loc=0, scale=1, size=1000)
dist2 = np.random.normal(loc=1, scale=1, size=1000)

# Plot
sns.distplot(dist1, label='Distribution 1')
sns.distplot(dist2, label='Distribution 2')
plt.legend()
plt.show()
