# Widya Sari Wibowo (21091397070)

# insialisasi numpys
import numpy as np

# inisialisasi variable
# Input layer feature 10
# Per batchnya 6 input
inputs = [[1.0, 1.5, 2.0, 2.5, 3.3, 3.5, 4.9, 4.5, 5.0, 5.5],
          [1.7, 1.6, 2.4, 2.8, 3.4, 3.6, 4.4, 4.8, 5.2, 5.4],
          [9.2, 4.2, 1.3, 8.2, 2.4, 8.4, 5.8, 7.4, 1.6, 9.3],
          [2.8, 1.8, 2.6, 2.8, 3.6, 3.8, 4.6, 4.8, 5.6, 5.8],
          [2.5, 6.4, 7.2, 7.4, 8.2, 8.4, 7.2, 9.4, 1.2, 3.4],
          [2.3, 5.4, 2.4, 3.2, 3.4, 4.2, 6.4, 7.7, 8.2, 2.5]]

# inisialisasi bobot variable
# jumlah weight sesuai dengan jumlah neuron, yaitu 5
weights = [[0.5, 0.7, 1.4, 2.7, 7.8, 9.4, 3.2, 4.6, 7.3, 9.4],
           [1.5, 3.4, 0.9, 3.2, 0.4, 0.1, 2.8, 6.2, 8.4, 3.7],
           [2.7, 1.3, 1.4, 7.2, 9.8, 0.2, 6.5, 8.4, 5.3, 6.4],
           [6.1, 9.3, 4.2, 7.4, 0.3, 2.5, 1.3, 9.3, 8.2, 4.5],
           [2.7, 8.5, 0.2, 1.5, 3.2, 1.9, 0.8, 4.3, 6.4, 4.8]]

# inisialisasi bias
# jumlah bias dan jumlah neuron sama, yaitu 5
bias = [2.3, 3.5, 0.1, 1.5, 3.9]

# penghitungan ouput
outputs = np.dot(inputs, np.array(weights) . T) + bias

# mencetak output
print(outputs)
