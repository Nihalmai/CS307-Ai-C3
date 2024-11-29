import numpy as np

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        """Train the network using given patterns."""
        for pattern in patterns:
            bipolar_pattern = self._binary_to_bipolar(pattern)
            self.weights += np.outer(bipolar_pattern, bipolar_pattern)
        np.fill_diagonal(self.weights, 0)  # Avoid self-connections

    def recall(self, pattern, iterations=5):
        """Recall a pattern from the network."""
        state = self._binary_to_bipolar(pattern)
        for _ in range(iterations):
            for i in range(self.size):
                net_input = np.dot(self.weights[i], state)
                state[i] = 1 if net_input > 0 else -1
        return self._bipolar_to_binary(state)

    def check_capacity(self, patterns):
        """Evaluate how well the network recalls patterns."""
        self.train(patterns)
        correct_recalls = sum(
            np.array_equal(self.recall(pattern), pattern) for pattern in patterns
        )
        return correct_recalls / len(patterns)

    def _binary_to_bipolar(self, pattern):
        """Convert binary pattern to bipolar (-1, 1)."""
        return 2 * pattern - 1

    def _bipolar_to_binary(self, pattern):
        """Convert bipolar pattern back to binary (0, 1)."""
        return (pattern + 1) // 2

# Parameters
network_size = 100  # Network size (10x10 grid)
estimated_capacity = int(0.15 * network_size)  # Estimated capacity

# Create random binary patterns
patterns = [np.random.randint(0, 2, network_size) for _ in range(estimated_capacity)]

# Assess the network's capacity
hopfield_net = HopfieldNetwork(network_size)
recall_accuracy = hopfield_net.check_capacity(patterns)

print(f"Estimated Capacity (P_max): {estimated_capacity} patterns")
print(f"Recall Accuracy with {estimated_capacity} patterns: {recall_accuracy * 100:.2f}%")
