import csv
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns


path = r'./pnp.csv'


calc = []
real = []

with open(path, 'r', encoding='gb2312') as f:
    lines = csv.reader(f)
    next(f)
    for line in lines:
        calc.append(float(line[0]))
        real.append(float(line[1]))

# def fit(x, A, B, C):
#     return (A * x*x + B * x + C)

def fit(x, A, B):
    return (A * x + B)

# A1, B1, C1 = optimize.curve_fit(fit, calc, real)[0]
# print('pnp: real = {:.2f}calc^2 + {:.2f}calc + {:.2f}'.format(A1, B1, C1))

A1, B1 = optimize.curve_fit(fit, calc, real)[0]
print('pnp: real = {:.2f}calc + {:.2f}'.format(A1, B1))

# sns.set(style="white", palette="muted", color_codes=True)
# distance = np.arange(1, 5, .2)
# drop1 = np.array([round(A1*d*d+B1*d+C1, 2) for d in distance])

# # Set up the matplotlib figure
# sns.despine(left=True)
# sns.scatterplot(distance, distance)
# sns.scatterplot(distance, drop1)

# plt.tight_layout()

# plt.show()