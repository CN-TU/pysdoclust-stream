import matplotlib.pyplot as plt
import numpy as np

# # Define the starting color (blue) and the target color (orange)
# start_color = np.array([0, 80, 239]) / 255  # Hex: #0050EF
# end_color = np.array([250, 104, 0]) / 255  # Hex: "#FA6800"

# # Create a color gradient
# steps = 12
# gradient = np.linspace(start_color, end_color, steps)

import numpy as np

# Create linspace from sqrt(1.1) to sqrt(10) with 20 entries
linspace_sqrt = np.linspace(np.sqrt(1.1), np.sqrt(10), 20)

# Square the values
squared_values = linspace_sqrt ** 2

print(squared_values)