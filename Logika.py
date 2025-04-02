import json
import webview
from collections import deque

def logika(cost_matrix, supply, demand, purchase_prices, selling_prices):
    rows, cols = len(cost_matrix), len(cost_matrix[0])
    supply_left = supply.copy()
    demand_left = demand.copy()

    # Store original data
    original_cost_matrix = [row[:] for row in cost_matrix]
    original_supply = supply.copy()
    original_demand = demand.copy()
    original_purchase_prices = purchase_prices.copy()
    original_selling_prices = selling_prices.copy()

    # Handle imbalance
    total_supply = sum(supply)
    total_demand = sum(demand)
    if total_supply > total_demand:
        print(f"Supply ({total_supply}) exceeds demand ({total_demand}). Adding dummy receiver.")
        demand_left.append(total_supply - total_demand)
        for i in range(rows):
            cost_matrix[i].append(0)
        selling_prices.append(0)
        cols += 1
    elif total_demand > total_supply:
        print(f"Demand ({total_demand}) exceeds supply ({total_supply}). Adding dummy supplier.")
        supply_left.append(total_demand - total_supply)
        cost_matrix.append([0] * cols)
        purchase_prices.append(0)
        rows += 1

    # Calculate profit matrix
    profit_matrix = []
    for i in range(rows):
        row = []
        for j in range(cols):
            profit = selling_prices[j] - purchase_prices[i] - cost_matrix[i][j]
            row.append(profit)
        profit_matrix.append(row)

    # Initial allocation: maximum profit method
    allocation = [[0] * cols for _ in range(rows)]
    temp_supply = supply_left.copy()
    temp_demand = demand_left.copy()
    while any(temp_supply) and any(temp_demand):
        max_profit = float('-inf')
        max_i, max_j = -1, -1
        for i in range(rows):
            for j in range(cols):
                if temp_supply[i] > 0 and temp_demand[j] > 0 and profit_matrix[i][j] > max_profit:
                    max_profit = profit_matrix[i][j]
                    max_i, max_j = i, j
        if max_i == -1 or max_j == -1:
            break
        amount = min(temp_supply[max_i], temp_demand[max_j])
        allocation[max_i][max_j] = amount
        temp_supply[max_i] -= amount
        temp_demand[max_j] -= amount

    initial_allocation = [row[:] for row in allocation]

    # Handle degeneracy
    epsilon = 0.001
    occupied_cells = sum(1 for i in range(rows) for j in range(cols) if allocation[i][j] > 0)
    required_cells = rows + cols - 1
    if occupied_cells < required_cells:
        for i in range(rows):
            for j in range(cols):
                if allocation[i][j] == 0 and occupied_cells < required_cells:
                    allocation[i][j] = epsilon
                    occupied_cells += 1
        print(f"Added epsilon ({epsilon}) for degeneracy.")

    # Optimization using potential method
    def calculate_potentials(allocation):
        u = [None] * rows
        v = [None] * cols
        u[rows - 1] = 0  # Start with dummy supplier
        cells_to_solve = [(i, j) for i in range(rows) for j in range(cols) if allocation[i][j] > 0]
        while cells_to_solve:
            progress = False
            for i, j in cells_to_solve[:]:
                if u[i] is not None and v[j] is None:
                    v[j] = profit_matrix[i][j] - u[i]
                    cells_to_solve.remove((i, j))
                    progress = True
                elif v[j] is not None and u[i] is None:
                    u[i] = profit_matrix[i][j] - v[j]
                    cells_to_solve.remove((i, j))
                    progress = True
            if not progress:
                break
        for i in range(rows):
            if u[i] is None:
                u[i] = 0
        for j in range(cols):
            if v[j] is None:
                v[j] = 0
        return u, v

    max_iterations = 100
    iterations = 0
    while iterations < max_iterations:
        u, v = calculate_potentials(allocation)
        improvement_possible = False
        entering_cell = None
        max_delta = float('-inf')
        delta_values = [[0] * cols for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                delta = profit_matrix[i][j] - (u[i] + v[j])
                delta_values[i][j] = delta
                if allocation[i][j] <= epsilon and delta > 0 and delta > max_delta:
                    max_delta = delta
                    entering_cell = (i, j)
                    improvement_possible = True
        if not improvement_possible:
            print(f"Optimal solution found after {iterations} iterations.")
            break

        start_i, start_j = entering_cell
        def find_path(start_i, start_j, epsilon_val):
            queue = deque([(start_i, start_j, '+', [(start_i, start_j, '+')], float('inf'))])
            visited = set()
            while queue:
                i, j, direction, path, theta = queue.popleft()
                if (i, j) in visited and (i, j) == (start_i, start_j) and len(path) > 2 and len(path) % 2 == 0:
                    theta = min(theta, min(allocation[r][c] for r, c, d in path if d == '-' and allocation[r][c] > epsilon_val))
                    return path, theta
                visited.add((i, j))
                if allocation[i][j] <= epsilon_val and (i, j) != (start_i, start_j):
                    continue
                if direction == '+':
                    for next_i in range(rows):
                        if allocation[next_i][j] > epsilon_val and (next_i, j) not in visited:
                            new_theta = min(theta, allocation[next_i][j])
                            queue.append((next_i, j, '-', path + [(next_i, j, '-')], new_theta))
                else:
                    for next_j in range(cols):
                        if allocation[i][next_j] > epsilon_val and (i, next_j) not in visited:
                            new_theta = min(theta, allocation[i][next_j])
                            queue.append((i, next_j, '+', path + [(i, next_j, '+')], new_theta))
            return None, 0

        path, theta = find_path(start_i, start_j, epsilon)
        if not path:
            print(f"No improving path for ({start_i}, {start_j}). Assuming optimal.")
            break

        print(f"Iteration {iterations + 1}: Improving path: {path}, theta: {theta}")
        for i, j, direction in path:
            if direction == '+':
                allocation[i][j] += theta
            else:
                allocation[i][j] -= theta

        for i in range(rows):
            for j in range(cols):
                if 0 < allocation[i][j] <= epsilon:
                    allocation[i][j] = 0

        iterations += 1
        print(f"Updated allocation: {allocation}")

    if iterations >= max_iterations:
        print("Max iterations reached.")

    # Financial calculations
    transport_cost = sum(original_cost_matrix[i][j] * allocation[i][j] for i in range(len(original_supply)) for j in range(len(original_demand)))
    purchase_cost = sum(sum(allocation[i][j] for j in range(len(original_demand))) * original_purchase_prices[i] for i in range(len(original_supply)))
    income = sum(sum(allocation[i][j] for i in range(len(original_supply))) * original_selling_prices[j] for j in range(len(original_demand)))
    total_cost = transport_cost + purchase_cost
    profit = sum(profit_matrix[i][j] * allocation[i][j] for i in range(len(original_supply)) for j in range(len(original_demand)))

    # Final allocation (no dummies)
    final_allocation = [row[:len(original_demand)] for row in allocation[:len(original_supply)]]

    steps = {
        "initial_allocation": [row[:len(original_demand)] for row in initial_allocation[:len(original_supply)]],
        "allocation": final_allocation,
        "total_cost": total_cost,
        "transport_cost": transport_cost,
        "purchase_cost": purchase_cost,
        "income": income,
        "profit": profit,
        "potentials": {"u": u[:len(original_supply)], "v": v[:len(original_demand)]},
        "iterations": iterations,
        "improvement_possible": iterations > 0
    }
    return final_allocation, total_cost, steps

def calculate(costs, supply, demand, purchase_prices, selling_prices):
    allocation, total_cost, steps = logika(costs, supply, demand, purchase_prices, selling_prices)
    return steps

if __name__ == "__main__":
    # Test case from PDF (part a)
    costs = [[8, 14, 17], [12, 9, 19]]
    supply = [20, 30]
    demand = [10, 28, 27]
    purchase_prices = [10, 12]
    selling_prices = [30, 25, 30]
    allocation, cost, steps = logika(costs, supply, demand, purchase_prices, selling_prices)
    print(f"Test Case 1 (PDF part a):")
    print(f"Solution: {allocation}")
    print(f"Transport Cost: {steps['transport_cost']}")
    print(f"Purchase Cost: {steps['purchase_cost']}")
    print(f"Income: {steps['income']}")
    print(f"Profit: {steps['profit']}")
    print(f"Steps: {steps}")

    # Test case from PDF (part d)
    costs_d = [[8, 14, 17], [12, 9, 19]]
    supply_d = [20, 30]
    demand_d = [10, 28, 27]  # O3 must be fully satisfied (27)
    purchase_prices_d = [10, 12]
    selling_prices_d = [30, 25, 30]
    allocation_d, cost_d, steps_d = logika(costs_d, supply_d, demand_d, purchase_prices_d, selling_prices_d)
    print(f"\nTest Case 2 (PDF part d):")
    print(f"Solution: {allocation_d}")
    print(f"Transport Cost: {steps_d['transport_cost']}")
    print(f"Purchase Cost: {steps_d['purchase_cost']}")
    print(f"Income: {steps_d['income']}")
    print(f"Profit: {steps_d['profit']}")
    print(f"Steps: {steps_d}")

    # Webview
    api = type('API', (), {
        'calculate': lambda costs, supply, demand, purchase_prices, selling_prices: calculate(
            json.loads(costs),
            json.loads(supply),
            json.loads(demand),
            json.loads(purchase_prices),
            json.loads(selling_prices)
        )
    })()
    webview.create_window("Intermediary Problem", "Index.html", js_api=api)
    webview.start()