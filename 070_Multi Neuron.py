# Widya Sari Wibowo (21091397070)

# inisialisasi numpy
import numpy as np

# inisialisasi variable
# input layer feature 10
inputs = [3, 5, 4, 4.5, 2, 6, 3.5, 8, 9, 1]

# inisialisasi bobot variable
# jumlah weight sesuai dengan jumlah neuron, yaitu 5
weights = [[-3.4, 1.4, 0.8, 2.9, 2.4, -1.6, 0.6, 2.2, 5.6, 1.5],
           [1.5, 3.2, 4.5, 7.8, -2.3, 6.7, 8.3, 9.7, -1.2, 4.5],
           [0.1, 0.3, 0.7, 0.9, 1.0, 1.1, 1.3, 1.7, 1.9, 2.0],
           [2.4, 1.8, 2.6, 2.8, 3.6, 3.8, 4.6, 4.8, 5.6, 5.8],
           [6.0, 6.2, 6.6, 6.8, 7.0, 7.2, 7.4, 7.6, 7.8, 8.0]]

# inisialisasi bias
# jumlah bias dan jumlah neuron sama, yaitu 5
bias = [1.4, 4, 2.7, -5.4, 7.2]

# penghitungan output
output = np.dot(weights, inputs) + bias

# mencetak output
print(output)
