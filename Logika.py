import json
import webview
from collections import deque

def logika(cost_matrix, supply, demand, purchase_prices, selling_prices):
    rows, cols = len(cost_matrix), len(cost_matrix[0])

    # Kopie danych
    supply_left = supply.copy()
    demand_left = demand.copy()
    original_cost_matrix = [row[:] for row in cost_matrix]
    original_supply = supply.copy()
    original_demand = demand.copy()
    original_purchase_prices = purchase_prices.copy()
    original_selling_prices = selling_prices.copy()

    # Sprawdzenie równowagi podaży i popytu
    total_supply = sum(supply)
    total_demand = sum(demand)

    # Obsługa niezrównoważonego problemu
    if total_supply > total_demand:
        print(f"Suma podaży ({total_supply}) jest większa niż suma popytu ({total_demand}). Dodajemy fikcyjnego odbiorcę.")
        demand_left.append(total_supply - total_demand)
        for i in range(rows):
            cost_matrix[i].append(0)  # Koszt transportu do fikcyjnego odbiorcy = 0
        selling_prices.append(0)  # Cena sprzedaży dla fikcyjnego odbiorcy = 0
        cols += 1
    elif total_demand > total_supply:
        print(f"Suma popytu ({total_demand}) jest większa niż suma podaży ({total_supply}). Dodajemy fikcyjnego dostawcę.")
        supply_left.append(total_demand - total_supply)
        cost_matrix.append([0] * cols)  # Koszt transportu od fikcyjnego dostawcy = 0
        purchase_prices.append(0)  # Cena zakupu dla fikcyjnego dostawcy = 0
        rows += 1

    # Po modyfikacji suma podaży i popytu powinna być równa
    total_supply = sum(supply_left)
    total_demand = sum(demand_left)
    if total_supply != total_demand:
        raise ValueError(f"Po modyfikacji suma podaży ({total_supply}) musi równać się sumie popytu ({total_demand})!")

    # Obliczanie zysku na jednostkę i przekształcenie na problem minimalizacji
    profit_matrix = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            # Zysk = cena sprzedaży - cena zakupu - koszt transportu
            profit = selling_prices[j] - purchase_prices[i] - cost_matrix[i][j]
            # Przekształcamy maksymalizację zysku w minimalizację kosztu: cost = -profit
            profit_matrix[i][j] = -profit

    # Inicjalizacja macierzy alokacji
    allocation = [[0] * cols for _ in range(rows)]

    # Metoda maksymalnego zysku (minimalizacja negatywnego zysku)
    while any(supply_left) and any(demand_left):
        max_profit = float('-inf')  # Szukamy największego zysku (najmniejszego negatywnego kosztu)
        min_i, min_j = -1, -1
        for i in range(rows):
            for j in range(cols):
                if supply_left[i] > 0 and demand_left[j] > 0 and profit_matrix[i][j] > max_profit:
                    max_profit = profit_matrix[i][j]
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
        epsilon = 0.001
        for i in range(rows):
            for j in range(cols):
                if allocation[i][j] == 0 and occupied_cells < required_cells:
                    allocation[i][j] = epsilon
                    occupied_cells += 1
        print(f"Dodano sztuczne alokacje (epsilon={epsilon}) do rozwiązania zdegenerowanego.")

    initial_allocation = [row[:] for row in allocation]

    # Funkcja obliczania potencjałów
    def calculate_potentials(allocation):
        u = [None] * rows
        v = [None] * cols
        u[0] = 0  # Punkt odniesienia
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

        # Wypełnienie pozostałych wartości zerami
        for i in range(rows):
            if u[i] is None:
                u[i] = 0
        for j in range(cols):
            if v[j] is None:
                v[j] = 0
        print(f"Potencjały: u={u}, v={v}")
        return u, v

    # Optymalizacja metodą potencjałów
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
                    shadow_cost = profit_matrix[i][j] - (u[i] + v[j])
                    if shadow_cost < 0 and shadow_cost < max_saving:
                        max_saving = shadow_cost
                        entering_cell = (i, j)
                        improvement_possible = True

        if not improvement_possible:
            print(f"Brak możliwości poprawy po {iterations} iteracjach.")
            break

        start_i, start_j = entering_cell

        # Znajdowanie ścieżki za pomocą BFS
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

    # Obliczanie wyników finansowych (tylko dla oryginalnych dostawców i odbiorców)
    transport_cost = 0
    purchase_cost = 0
    income = 0

    # Koszt transportu
    for i in range(len(original_supply)):
        for j in range(len(original_demand)):
            transport_cost += original_cost_matrix[i][j] * allocation[i][j]

    # Koszt zakupu
    for i in range(len(original_supply)):
        total_delivered_from_supplier = sum(allocation[i][j] for j in range(len(original_demand)))
        purchase_cost += total_delivered_from_supplier * original_purchase_prices[i]

    # Przychód
    for j in range(len(original_demand)):
        total_delivered_to_customer = sum(allocation[i][j] for i in range(len(original_supply)))
        income += total_delivered_to_customer * original_selling_prices[j]

    # Całkowity koszt i zysk
    total_cost = transport_cost + purchase_cost
    profit = income - total_cost

    # Przygotowanie alokacji do zwrotu (usuwamy fikcyjnego dostawcę/odbiorcę)
    final_allocation = [row[:len(original_demand)] for row in allocation[:len(original_supply)]]
    initial_allocation = [row[:len(original_demand)] for row in initial_allocation[:len(original_supply)]]

    steps = {
        "initial_allocation": initial_allocation,
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

# Interfejs pywebview
def calculate(costs, supply, demand, purchase_prices, selling_prices):
    allocation, total_cost, steps = logika(costs, supply, demand, purchase_prices, selling_prices)
    return steps

if __name__ == "__main__":
    # Dane z zadania 3 z PDF-a
    costs = [[5, 3, 8], [9, 2, 4]]  # Koszty transportu
    supply = [20, 30]  # Podaż D1, D2
    demand = [18, 32, 20]  # Popyt O1, O2, O3
    purchase_prices = [6, 9]  # Ceny zakupu od D1, D2
    selling_prices = [15, 14, 16]  # Ceny sprzedaży do O1, O2, O3

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