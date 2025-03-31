def logika(cost_matrix, supply, demand, purchase_prices, selling_prices):
    rows, cols = len(cost_matrix), len(cost_matrix[0])
    allocation = [[0] * cols for _ in range(rows)]
    supply_left = supply.copy()
    demand_left = demand.copy()

    # Sprawdzenie równowagi podaży i popytu
    total_supply = sum(supply)
    total_demand = sum(demand)
    if total_supply != total_demand:
        raise ValueError(f"Suma podaży ({total_supply}) musi równać się sumie popytu ({total_demand})!")

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

    # Sprawdzenie degeneracji
    occupied_cells = sum(1 for i in range(rows) for j in range(cols) if allocation[i][j] > 0)
    required_cells = rows + cols - 1
    if occupied_cells < required_cells:
        # Dodajemy sztuczne alokacje o wartości epsilon
        epsilon = 0.001
        for i in range(rows):
            for j in range(cols):
                if allocation[i][j] == 0 and occupied_cells < required_cells:
                    allocation[i][j] = epsilon
                    occupied_cells += 1
        print(f"Dodano sztuczne alokacje (epsilon={epsilon}) do rozwiązania zdegenerowanego.")

    initial_allocation = [row[:] for row in allocation]

    def calculate_potentials(allocation):
        u = [None] * rows
        v = [None] * cols
        u[0] = 0  # Punkt odniesienia
        cells_to_solve = [(i, j) for i in range(rows) for j in range(cols) if allocation[i][j] > 0]
        
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
            if not progress:
                break
        
        # Wypełnienie pozostałych wartości zerami
        for i in range(rows):
            if u[i] is None:
                u[i] = 0
        for j in range(cols):
            if v[j] is None:
                v[j] = 0
        
        print(f"Potencjały: u={u}, v={v}")
        return u, v

    max_iterations = 100
    iterations = 0

    while iterations < max_iterations:
        u, v = calculate_potentials(allocation)
        improvement_possible = False
        entering_cell = None
        max_saving = float('-inf')

        # Sprawdzanie możliwości poprawy
        for i in range(rows):
            for j in range(cols):
                if allocation[i][j] == 0:
                    shadow_cost = cost_matrix[i][j] - (u[i] + v[j])
                    if shadow_cost < 0 and shadow_cost < max_saving:
                        max_saving = shadow_cost
                        entering_cell = (i, j)
                        improvement_possible = True

        if not improvement_possible:
            print(f"Brak możliwości poprawy po {iterations} iteracjach.")
            break

        start_i, start_j = entering_cell

        # Zoptymalizowane znajdowanie ścieżki za pomocą BFS
        from collections import deque
        def find_path(start_i, start_j):
            queue = deque([(start_i, start_j, '+', [(start_i, start_j, '+')], float('inf'))])
            visited = set()
            while queue:
                i, j, direction, current_path, theta = queue.popleft()
                if (i, j) in visited and (i, j) == (start_i, start_j) and len(current_path) > 2 and len(current_path) % 2 == 0:
                    theta = min(theta, min(allocation[r][c] for r, c, d in current_path if d == '-' and allocation[r][c] > epsilon))
                    return current_path, theta
                visited.add((i, j))
                if allocation[i][j] <= epsilon and (i, j) != (start_i, start_j):
                    continue
                if direction == '+':
                    for next_i in range(rows):
                        if allocation[next_i][j] > epsilon and (next_i, j) not in visited:
                            new_theta = min(theta, allocation[next_i][j])
                            queue.append((next_i, j, '-', current_path + [(next_i, j, '-')], new_theta))
                else:
                    for next_j in range(cols):
                        if allocation[i][next_j] > epsilon and (i, next_j) not in visited:
                            new_theta = min(theta, allocation[i][next_j])
                            queue.append((i, next_j, '+', current_path + [(i, next_j, '+')], new_theta))
            return None, 0

        path, theta = find_path(start_i, start_j)
        if not path:
            print(f"Nie znaleziono zamkniętej ścieżki dla ({start_i}, {start_j}). Przerwanie.")
            break

        print(f"Ścieżka: {path}, theta: {theta}")
        for i, j, direction in path:
            if direction == '+':
                allocation[i][j] += theta
            else:
                allocation[i][j] -= theta

        # Usuwanie epsilonów po aktualizacji
        for i in range(rows):
            for j in range(cols):
                if 0 < allocation[i][j] <= epsilon:
                    allocation[i][j] = 0

        iterations += 1
        print(f"Iteracja {iterations}, alokacja: {allocation}")

    if iterations >= max_iterations:
        print("Osiągnięto maksymalną liczbę iteracji.")

    # Obliczanie kosztu transportu
    transport_cost = sum(cost_matrix[i][j] * allocation[i][j] for i in range(rows) for j in range(cols))

    # Obliczanie kosztu zakupu
    # Koszt zakupu = suma (ilość dostarczona od dostawcy i * cena zakupu dla dostawcy i)
    purchase_cost = 0
    for i in range(rows):
        total_delivered_from_supplier = sum(allocation[i][j] for j in range(cols))
        purchase_cost += total_delivered_from_supplier * purchase_prices[i]

    # Całkowity koszt = koszt transportu + koszt zakupu
    total_cost = transport_cost + purchase_cost

    # Obliczanie przychodu (income)
    # Przychód = suma (ilość dostarczona do klienta j * cena sprzedaży dla klienta j)
    income = 0
    for j in range(cols):
        total_delivered_to_customer = sum(allocation[i][j] for i in range(rows))
        income += total_delivered_to_customer * selling_prices[j]

    # Obliczanie zysku (profit)
    profit = income - total_cost

    steps = {
        "initial_allocation": initial_allocation,
        "allocation": allocation,
        "total_cost": total_cost,
        "transport_cost": transport_cost,  # Dodajemy dla informacji
        "purchase_cost": purchase_cost,    # Dodajemy dla informacji
        "income": income,
        "profit": profit,
        "potentials": {"u": u, "v": v},
        "iterations": iterations,
        "improvement_possible": iterations > 0
    }
    
    return allocation, total_cost, steps

# Interfejs pywebview
import json
import webview

def calculate(costs, supply, demand, purchase_prices, selling_prices):
    allocation, total_cost, steps = logika(costs, supply, demand, purchase_prices, selling_prices)
    return steps

if __name__ == "__main__":
    # Testowy przykład
    costs = [[4, 6], [5, 7]]
    supply = [50, 60]
    demand = [60, 50]
    purchase_prices = [2, 3]  # Przykładowe ceny zakupu dla dostawców
    selling_prices = [12, 13]  # Przykładowe ceny sprzedaży
    allocation, cost, steps = logika(costs, supply, demand, purchase_prices, selling_prices)
    print(f"Rozwiązanie: {allocation}")
    print(f"Całkowity koszt: {cost}")
    print(f"Kroki: {steps}")

    # Uruchomienie interfejsu
    api = type('API', (), {
        'calculate': lambda costs, supply, demand, purchase_prices, selling_prices: calculate(
            json.loads(costs), 
            json.loads(supply), 
            json.loads(demand), 
            json.loads(purchase_prices),
            json.loads(selling_prices)
        )
    })()

    webview.create_window("Metoda Pośrednika", "Index.html", js_api=api)

    webview.start()