# Logika.py

def logika(cost_matrix, supply, demand):
    # Inicjalizacja podstawowego rozwiązania (np. metodą minimalnego kosztu)
    rows, cols = len(cost_matrix), len(cost_matrix[0])
    allocation = [[0] * cols for _ in range(rows)]
    supply_left = supply.copy()
    demand_left = demand.copy()
    
    # Metoda minimalnego kosztu jako start
    for i in range(rows):
        for j in range(cols):
            if supply_left[i] > 0 and demand_left[j] > 0:
                amount = min(supply_left[i], demand_left[j])
                allocation[i][j] = amount
                supply_left[i] -= amount
                demand_left[j] -= amount
    
    # Główna pętla metody pośrednika
    while True:
        # Znajdź komórkę do poprawy (uproszczona logika)
        improvement_possible = False
        for i in range(rows):
            for j in range(cols):
                if allocation[i][j] == 0:
                    # Sprawdź czy wprowadzenie tej komórki poprawi rozwiązanie
                    # (tu powinna być pełna logika ścieżki zamkniętej)
                    pass
        
        if not improvement_possible:
            break
    
    # Oblicz całkowity koszt
    total_cost = sum(cost_matrix[i][j] * allocation[i][j] 
                    for i in range(rows) 
                    for j in range(cols))
    
    return allocation, total_cost

# Przykład użycia
if __name__ == "__main__":
    costs = [
        [4, 6, 8],
        [5, 7, 2],
        [8, 6, 5]
    ]
    supply = [50, 60, 40]
    demand = [60, 50, 40]
    
    solution, cost = logika(costs, supply, demand)
    print(f"Rozwiązanie: {solution}")
    print(f"Całkowity koszt: {cost}")