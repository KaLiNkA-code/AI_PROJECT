import numpy as np
from matplotlib import pyplot as plt

x_train = np.array([1.0, 3.0, 1.5, 1.8, 2.5, 2.8])
y_train = np.array([300, 500, 350, 380, 420, 475])

m = len(x_train)  # m is the number of training examples
print(f"number of training examples: {m}")

print(" ")
print(" ")
print(" ")


i = 0
x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")


print(" ")
print(" ")
print(" ")

plt.scatter(x_train, y_train, marker="x", c="r")
plt.title("Housing Prices")  # Set the title
plt.ylabel("Price (in 1000s of dollars)")
plt.xlabel("Size (1000 sqrt)")
# plt.show()

w = 100
b = 193
print(f"w: {w}")
print(f"b: {b}")


def calculate_model_output(w, b, x):
    m = x.shape  # the number of training examples
    f_wb = np.zeros(m)
    for i in range(len(x)):
        f_wb[i] = w * x[i] + b
    return f_wb


#  Now Let's call the calculate_model_output func and plot the output
tmp_f_wb = calculate_model_output(w, b, x_train)

plt.plot(x_train, tmp_f_wb, c="b", label="Our Prediction")
plt.scatter(x_train, y_train, marker="x", c="r", label="Actual price")
plt.xlabel("Sixe")
plt.ylabel("Price")
plt.legend()
plt.show()

print( 'hello world' )
print( 'test to refactoring pre-commit' )
