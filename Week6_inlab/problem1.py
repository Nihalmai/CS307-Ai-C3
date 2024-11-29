import numpy as np
import matplotlib.pyplot as plt

# In the visualization, 1 represents an empty space and 0 represents a filled square

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))  # Initialize weights to zero

    def train(self, patterns):
        """Train the network with provided binary patterns."""
        for pattern in patterns:
            # Convert binary pattern to bipolar (-1, 1)
            bipolar_pattern = 2 * pattern - 1
            self.weights += np.outer(bipolar_pattern, bipolar_pattern)
        
        # No self-connections
        np.fill_diagonal(self.weights, 0)

    def recall(self, pattern, iterations=5):
        """Recall a pattern using asynchronous updates."""
        current_state = pattern.copy()
        for _ in range(iterations):
            for i in range(self.size):
                net_input = np.dot(self.weights[i], current_state)
                current_state[i] = 1 if net_input > 0 else -1
        return (current_state + 1) // 2  # Convert back to binary (0, 1)

    def energy(self, state):
        """Calculate the energy of a given state."""
        bipolar_state = 2 * state - 1
        return -0.5 * np.dot(bipolar_state.T, np.dot(self.weights, bipolar_state))

# Create a 10x10 Hopfield network
network_size = 100  # Size for a 10x10 grid
hopfield_net = HopfieldNetwork(network_size)

# Define binary patterns (each as a 10x10 grid flattened to a 1D array)
binary_patterns = [
    np.random.randint(0, 2, network_size),  # Random pattern
    np.random.randint(0, 2, network_size),  # Another random pattern
]

# Train the network with the patterns
hopfield_net.train(binary_patterns)

# Test the network with a noisy version of the first pattern
test_pattern = binary_patterns[0].copy()
# Introduce noise by flipping some bits
noise_indices = np.random.choice(network_size, network_size // 10, replace=False)
test_pattern[noise_indices] = 1 - test_pattern[noise_indices]

# Recall the pattern from the network
recalled_pattern = hopfield_net.recall(test_pattern)

# Function to display and print patterns
def display_and_print_pattern(pattern, title):
    """Show the pattern as an image and print it as a matrix."""
    pattern_matrix = pattern.reshape(10, 10)
    print(f"{title} (Binary Matrix):")
    print(pattern_matrix)
    print("\n")
    plt.imshow(pattern_matrix, cmap="gray", interpolation="nearest")
    plt.title(title)
    plt.axis("off")

# Plot and print the patterns
plt.figure(figsize=(10, 4))

# Original pattern
plt.subplot(1, 3, 1)
display_and_print_pattern(binary_patterns[0], "Original Pattern")

# Noisy input pattern
plt.subplot(1, 3, 2)
display_and_print_pattern(test_pattern, "Noisy Input Pattern")

# Recalled pattern
plt.subplot(1, 3, 3)
display_and_print_pattern(recalled_pattern, "Recalled Pattern")

plt.tight_layout()
plt.show()
