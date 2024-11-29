import numpy as np
import matplotlib.pyplot as plt
import random

# Define binary patterns for letters D, J, C, M
letter_patterns = {
    'D': np.array([-1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1]).reshape(5, 5),
    'J': np.array([-1, -1, -1, -1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1]).reshape(5, 5),
    'C': np.array([1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1]).reshape(5, 5),
    'M': np.array([-1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1]).reshape(5, 5)
}

# Visualize the patterns
plt.figure(figsize=(6, 6))
for index, (letter, pattern) in enumerate(letter_patterns.items()):
    plt.subplot(1, 4, index + 1)
    plt.title(letter)
    plt.axis('off')
    plt.imshow(pattern, cmap='gray')
plt.show()

# Hebbian learning to train the Hopfield network
num_patterns = len(letter_patterns)
pattern_dim = np.prod(next(iter(letter_patterns.values())).shape)
weight_matrix = np.zeros((pattern_dim, pattern_dim))

for pattern in letter_patterns.values():
    flat_pattern = pattern.flatten()
    weight_matrix += np.outer(flat_pattern, flat_pattern)

np.fill_diagonal(weight_matrix, 0)
weight_matrix /= num_patterns

def add_noise(pattern, num_errors=1):
    """Introduce errors to a pattern."""
    noisy_pattern = pattern.copy()
    error_positions = set()
    while len(error_positions) < num_errors:
        i, j = np.random.randint(0, 5, size=2)
        if (i, j) not in error_positions:
            error_positions.add((i, j))
            noisy_pattern[i, j] *= -1
    return noisy_pattern

# Error correction visualization
plt.figure(figsize=(9, 45))
for error_count in range(1, 11):
    chosen_letter = random.choice(list(letter_patterns.values()))
    noisy_letter = add_noise(chosen_letter, error_count)
    y = noisy_letter.flatten()

    # Iteratively correct errors
    prev_norm = float('inf')
    curr_norm = np.linalg.norm(y)
    while curr_norm < prev_norm:
        prev_norm = curr_norm
        y = np.sign(weight_matrix @ y)
        curr_norm = np.linalg.norm(y - noisy_letter.flatten())

    # Plot results
    plt.subplot(10, 3, 3 * (error_count - 1) + 1)
    plt.title('Original')
    plt.axis('off')
    plt.imshow(chosen_letter, cmap='gray')

    plt.subplot(10, 3, 3 * (error_count - 1) + 2)
    plt.title(f'With {error_count} error(s)')
    plt.axis('off')
    plt.imshow(noisy_letter, cmap='gray')

    plt.subplot(10, 3, 3 * (error_count - 1) + 3)
    plt.title('Corrected')
    plt.axis('off')
    plt.imshow(y.reshape(5, 5), cmap='gray')

plt.show()
