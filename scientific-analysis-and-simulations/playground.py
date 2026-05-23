import numpy as np

arr1 = np.zeros(10)
arr2 = np.array([1,1,1])

# Use np.nan as the "pass" value
conditions = np.array([5, np.nan])
indices = [0, 1]

# Elegant one-liner: "If not NaN, use condition, else keep existing value"
# This reads: For the target indices, use 'conditions' if it's valid, otherwise keep 'arr1' as is.
arr1[indices] = np.where(~np.isnan(conditions), conditions, arr1[indices])

arr3 = np.array([2,1,3])
arr4 = np.array([1,2,3])
print(arr3 * arr4)