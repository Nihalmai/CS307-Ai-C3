import numpy as np
import matplotlib.pyplot as plt

def setup_board(size=8):
    """Create an initial board with randomly placed rooks."""
    board = np.zeros((size, size), dtype=int)
    positions = np.random.choice(size * size, size, replace=False)
    for pos in positions:
        board[pos // size, pos % size] = 1
    return board

def assess_energy(board):
    """Evaluate the energy level of the board configuration."""
    row_collisions = np.sum((np.sum(board, axis=1) - 1) ** 2)
    col_collisions = np.sum((np.sum(board, axis=0) - 1) ** 2)
    return row_collisions + col_collisions

def enhance_board(board, iterations=1000):
    """Improve the board setup to minimize its energy."""
    current_energy = assess_energy(board)
    size = board.shape[0]
    
    for _ in range(iterations):
        pos1, pos2 = np.random.choice(size * size, 2, replace=False)
        r1, c1 = divmod(pos1, size)
        r2, c2 = divmod(pos2, size)
        
        if board[r1, c1] == 1 and board[r2, c2] == 0:
            board[r1, c1], board[r2, c2] = board[r2, c2], board[r1, c1]
            new_energy = assess_energy(board)
            if new_energy < current_energy:
                current_energy = new_energy
            else:
                board[r1, c1], board[r2, c2] = board[r2, c2], board[r1, c1]
    
    return board, current_energy

# Create and optimize the board
initial_board = setup_board()
plt.imshow(initial_board, cmap='binary', interpolation='nearest')
plt.title("Initial Board Setup")
plt.show()

optimized_board, final_energy = enhance_board(initial_board)
print("Final Energy of the Enhanced Board:", final_energy)

plt.imshow(optimized_board, cmap='binary', interpolation='nearest')
plt.title("Enhanced Board Setup")
plt.show()
