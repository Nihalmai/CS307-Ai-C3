import numpy as np
import matplotlib.pyplot as plt

# Generate random coordinates for cities
city_labels = [str(i) for i in range(1, 11)]
np.random.seed(0)  # Seed for reproducibility
coords = {label: np.random.rand(2) * 50 for label in city_labels}

# Function to create a distance matrix
def calculate_distances(city_labels):
    num_cities = len(city_labels)
    dist_matrix = np.zeros((num_cities, num_cities))
    for i, city1 in enumerate(city_labels):
        for j, city2 in enumerate(city_labels):
            if i != j:
                dist_matrix[i, j] = np.linalg.norm(coords[city1] - coords[city2])
    return dist_matrix

# Class to solve the Traveling Salesman Problem (TSP)
class TSP:
    def __init__(self, city_labels, dist_matrix):
        self.city_labels = city_labels
        self.num_cities = len(city_labels)
        self.dist_matrix = dist_matrix

    def tour_cost(self, tour):
        cost = sum(self.dist_matrix[tour[i], tour[i + 1]] for i in range(self.num_cities - 1))
        cost += self.dist_matrix[tour[-1], tour[0]]  # Return to starting point
        return cost

    def find_best_tour(self, iterations=100000):
        best_cost = float('inf')
        best_tour = None
        current_tour = np.random.permutation(self.num_cities)
        for _ in range(iterations):
            i, j = np.random.choice(self.num_cities, 2, replace=False)
            current_tour[i], current_tour[j] = current_tour[j], current_tour[i]
            current_cost = self.tour_cost(current_tour)
            if current_cost < best_cost:
                best_cost = current_cost
                best_tour = current_tour.copy()
        return best_tour, best_cost

# Compute the distance matrix
dist_matrix = calculate_distances(city_labels)

# Instantiate the TSP solver
tsp_solver = TSP(city_labels, dist_matrix)

# Solve the TSP
optimal_route, min_cost = tsp_solver.find_best_tour()

# Display the optimal route and its cost
print("Optimal Route:")
for i, city_index in enumerate(optimal_route):
    if i == 0:
        print(f"Start from {city_labels[city_index]}")
    else:
        print(f"Move to {city_labels[city_index]}")
print(f"Return to {city_labels[optimal_route[0]]}")
print("Minimum Travel Cost:", min_cost)

# Visualize the cities and the optimal route
plt.figure(figsize=(10, 8))
for city in city_labels:
    plt.scatter(*coords[city], color='green')
    plt.text(*coords[city], city, ha='center', va='center', fontsize=12)
for i in range(len(optimal_route) - 1):
    plt.plot([coords[city_labels[optimal_route[i]]][0], coords[city_labels[optimal_route[i + 1]]][0]],
             [coords[city_labels[optimal_route[i]]][1], coords[city_labels[optimal_route[i + 1]]][1]],
             color='blue')
plt.plot([coords[city_labels[optimal_route[-1]]][0], coords[city_labels[optimal_route[0]]][0]],
         [coords[city_labels[optimal_route[-1]]][1], coords[city_labels[optimal_route[0]]][1]],
         color='blue')

# Annotate the path distances
for i in range(len(optimal_route) - 1):
    mid_x = (coords[city_labels[optimal_route[i]]][0] + coords[city_labels[optimal_route[i + 1]]][0]) / 2
    mid_y = (coords[city_labels[optimal_route[i]]][1] + coords[city_labels[optimal_route[i + 1]]][1]) / 2
    plt.text(mid_x, mid_y, f'{dist_matrix[optimal_route[i], optimal_route[i + 1]]:.2f}', color='red')

plt.title("Traveling Salesman Problem Visualization")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.grid(True)
plt.show()
