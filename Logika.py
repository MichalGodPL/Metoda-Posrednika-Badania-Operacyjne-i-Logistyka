def logika(cost_matrix, supply, demand):
    rows, cols = len(cost_matrix), len(cost_matrix[0])
    allocation = [[0] * cols for _ in range(rows)]
    supply_left = supply.copy()
    demand_left = demand.copy()

    # Metoda minimalnego kosztu
    while any(supply_left) and any(demand_left):
        min_cost = float('inf')
        min_i, min_j = -1, -1
        for i in range(rows):
            for j in range(cols):
                if supply_left[i] > 0 and demand_left[j] > 0 and cost_matrix[i][j] < min_cost:
                    min_cost = cost_matrix[i][j]
                    min_i, min_j = i, j
        if min_i == -1 or min_j == -1:
            break
        amount = min(supply_left[min_i], demand_left[min_j])
        allocation[min_i][min_j] = amount
        supply_left[min_i] -= amount
        demand_left[min_j] -= amount

    # Obliczanie potencjałów
    def calculate_potentials(allocation):
        u = [None] * rows
        v = [None] * cols
        u[0] = 0  # Punkt odniesienia
        cells_to_solve = [(i, j) for i in range(rows) for j in range(cols) if allocation[i][j] > 0]
        
        # Iteracyjne wypełnianie potencjałów
        while cells_to_solve:
            progress = False
            for i, j in cells_to_solve[:]:
                if u[i] is not None and v[j] is None:
                    v[j] = cost_matrix[i][j] - u[i]
                    cells_to_solve.remove((i, j))
                    progress = True
                elif v[j] is not None and u[i] is None:
                    u[i] = cost_matrix[i][j] - v[j]
                    cells_to_solve.remove((i, j))
                    progress = True
            if not progress:  # Jeśli nie ma postępu, przerwij
                break
        
        # Wypełnij pozostałe None wartościami domyślnymi (np. 0)
        for i in range(rows):
            if u[i] is None:
                u[i] = 0
        for j in range(cols):
            if v[j] is None:
                v[j] = 0
        
        print(f"Potencjały: u={u}, v={v}")
        return u, v

    # Metoda pośrednika
    max_iterations = 100  # Zmniejszony limit dla szybszego testowania
    iteration = 0

    while iteration < max_iterations:
        u, v = calculate_potentials(allocation)
        improvement_possible = False
        entering_cell = None
        max_saving = float('-inf')

        for i in range(rows):
            for j in range(cols):
                if allocation[i][j] == 0:
                    shadow_cost = cost_matrix[i][j] - (u[i] + v[j])
                    if shadow_cost < 0 and shadow_cost < max_saving:
                        max_saving = shadow_cost
                        entering_cell = (i, j)
                        improvement_possible = True

        if not improvement_possible:
            print(f"Brak możliwości poprawy po {iteration} iteracjach.")
            break

        start_i, start_j = entering_cell
        path = []
        visited = set()

        def find_path(i, j, direction, amount=float('inf'), depth=0, max_depth=rows * cols):
            if depth > max_depth or (i, j) in visited:
                return False
            visited.add((i, j))
            if allocation[i][j] == 0 and (i, j) != (start_i, start_j):
                return False
            path.append((i, j, direction))
            
            if direction == '+':
                for next_i in range(rows):
                    if next_i != i and allocation[next_i][j] > 0:
                        amount = min(amount, allocation[next_i][j])
                        if find_path(next_i, j, '-', amount, depth + 1, max_depth):
                            return True
                path.pop()
            else:
                for next_j in range(cols):
                    if next_j != j and allocation[i][next_j] > 0:
                        amount = min(amount, allocation[i][next_j])
                        if find_path(i, next_j, '+', amount, depth + 1, max_depth):
                            return True
                path.pop()
            
            # Sprawdzamy, czy ścieżka się zamyka
            if (i, j) == (start_i, start_j) and len(path) > 2 and len(path) % 2 == 0:
                return True
            return False

        visited.clear()
        path.clear()
        found = find_path(start_i, start_j, '+')
        if not found or not path:
            print(f"Nie znaleziono zamkniętej ścieżki dla ({start_i}, {start_j}). Przerwanie.")
            break

        theta = float('inf')
        for p in path:
            if p[2] == '-':
                theta = min(theta, allocation[p[0]][p[1]])

        print(f"Ścieżka: {path}, theta: {theta}")
        for p in path:
            i, j, direction = p
            if direction == '+':
                allocation[i][j] += theta
            else:
                allocation[i][j] -= theta

        iteration += 1
        print(f"Iteracja {iteration}, alokacja: {allocation}")

    if iteration >= max_iterations:
        print("Osiągnięto maksymalną liczbę iteracji.")

    total_cost = sum(cost_matrix[i][j] * allocation[i][j] for i in range(rows) for j in range(cols))
    steps = {
        "initial_allocation": [row[:] for row in allocation],  # Początkowa alokacja to stan po metodzie min kosztu
        "final_allocation": allocation,
        "total_cost": total_cost,
        "potentials": {"u": u, "v": v}
    }
    
    return allocation, total_cost, steps

if __name__ == "__main__":
    costs = [[4, 6], [5, 7]]  # Test dla 2x2
    supply = [50, 60]
    demand = [60, 50]
    allocation, cost, steps = logika(costs, supply, demand)
    print(f"Rozwiązanie: {allocation}")
    print(f"Całkowity koszt: {cost}")
    print(f"Kroki: {steps}")