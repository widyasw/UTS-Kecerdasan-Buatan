# Widya Sari Wibowo (21091397070)

# insialisasi menggunakan numpy
import numpy as np

# inisialisasi variable
# input layer feature 10
inputs = [7, 2, 5.3, 4.5, 3, 6, 10, 8, 1, 9]

# inisialisasi bobot variable
# jumlah weight sesuai dengan jumlah neuron, yaitu 1
weights = [6.4, 2.8, 0.5, 5.3, 2.7, -2.5, 0.10, 5.6, 3.2, 1.8]

# inisialisasi bias
# jumlah bias sama dengan jumlah neuron, yaitu 1
bias = 4.0

# penghitungan output
output = np.dot(weights, inputs) + bias

# mencetak output
print(output)