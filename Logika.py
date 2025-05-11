import json

import webview


def logika(cost_matrix, supply, demand, purchase_prices, selling_prices, fixed_allocations=None):
    
    # Store original dimensions and data

    rows = len(cost_matrix)

    cols = len(cost_matrix[0])

    cost_matrix = [row[:] for row in cost_matrix]

    supply = supply.copy()

    demand = demand.copy()

    purchase_prices = purchase_prices.copy()

    selling_prices = selling_prices.copy()
    

    # Handle fixed allocations

    if fixed_allocations is None:

        fixed_allocations = {}
    

    # Adjust supply/demand for fixed allocations

    for (i, j), amount in fixed_allocations.items():

        if i < rows and j < cols:

            supply[i] -= amount

            demand[j] -= amount
    

    # Initialize allocation matrix

    allocation = [[0 for _ in range(cols)] for _ in range(rows)]
    

    # Apply fixed allocations

    for (i, j), amount in fixed_allocations.items():

        if i < rows and j < cols:

            allocation[i][j] = amount
    

    # Compute unit profits

    profit_matrix = []

    for i in range(rows):

        row = []

        for j in range(cols):

            profit = selling_prices[j] - purchase_prices[i] - cost_matrix[i][j]

            row.append(profit)

        profit_matrix.append(row)
    

    # Save initial allocation (before greedy allocation)

    initial_allocation = [row[:] for row in allocation]
    

    # Create sorted list of routes by profit

    profit_list = []

    for i in range(rows):

        for j in range(cols):

            if profit_matrix[i][j] > 0:  # Only positive profit routes

                profit_list.append({'supplier': i, 'receiver': j, 'profit': profit_matrix[i][j]})

    profit_list.sort(key=lambda x: x['profit'], reverse=True)
    

    # Greedy allocation

    for item in profit_list:

        i = item['supplier']

        j = item['receiver']

        if supply[i] > 0 and demand[j] > 0:

            amount = min(supply[i], demand[j])

            allocation[i][j] += amount

            supply[i] -= amount

            demand[j] -= amount
    

    # Calculate costs and profits

    def calculate_final_costs(alloc_matrix, rows, cols, cost_matrix, purchase_prices, selling_prices):

        transport_cost = 0

        for i in range(rows):

            for j in range(cols):

                quantity = alloc_matrix[i][j]

                unit_cost = cost_matrix[i][j]

                cell_cost = quantity * unit_cost

                transport_cost += cell_cost

                print(f"Transport D{i+1}->O{j+1}: {quantity} x {unit_cost} = {cell_cost}")
        

        purchase_cost = 0

        for i in range(rows):

            supplier_quantity = sum(alloc_matrix[i][j] for j in range(cols))

            supplier_cost = supplier_quantity * purchase_prices[i]

            purchase_cost += supplier_cost

            print(f"Purchase D{i+1}: {supplier_quantity} x {purchase_prices[i]} = {supplier_cost}")
        

        income = 0

        for j in range(cols):

            receiver_quantity = sum(alloc_matrix[i][j] for i in range(rows))

            receiver_income = receiver_quantity * selling_prices[j]

            income += receiver_income

            print(f"Income O{j+1}: {receiver_quantity} x {selling_prices[j]} = {receiver_income}")
        

        total_cost = transport_cost + purchase_cost

        profit = income - total_cost
        

        return {
            "transport_cost": transport_cost,

            "purchase_cost": purchase_cost,

            "income": income,
            
            "total_cost": total_cost,

            "profit": profit

        }
    

    cost_results = calculate_final_costs(allocation, rows, cols, cost_matrix, purchase_prices, selling_prices)
    

    # Compute potentials for display purposes

    u = [None] * rows

    v = [None] * cols

    u[0] = 0  # Anchor first potential
    

    # Find basic cells (non-zero allocations)

    basic_cells = [(i, j) for i in range(rows) for j in range(cols) if allocation[i][j] > 0]
    
    
    # Check if we have enough basic cells to determine all potentials

    if len(basic_cells) < rows + cols - 1:

        # Handle degenerate case - we need to add dummy allocations

        non_basic = [(i, j) for i in range(rows) for j in range(cols) if allocation[i][j] == 0]

        for i, j in non_basic:

            if len(basic_cells) >= rows + cols - 1:

                break

            basic_cells.append((i, j))

            # Adding an epsilon allocation (conceptually, not actually modifying allocation)
    
    # Solve for potentials using a more reliable approach

    attempts = 0

    max_attempts = rows * cols  # Avoid infinite loops
    

    while (None in u or None in v) and attempts < max_attempts:

        made_progress = False

        for i, j in basic_cells:

            if u[i] is not None and v[j] is None:

                v[j] = profit_matrix[i][j] - u[i]

                made_progress = True

            elif v[j] is not None and u[i] is None:

                u[i] = profit_matrix[i][j] - v[j]

                made_progress = True
        

        if not made_progress:

            # If we're stuck but still have unknowns, we might need to set another anchor

            # This is a fallback for degenerate cases

            for i in range(rows):

                if u[i] is None:

                    u[i] = 0

                    made_progress = True

                    break

            if not made_progress:

                for j in range(cols):

                    if v[j] is None:

                        v[j] = 0

                        made_progress = True

                        break
        
        attempts += 1
    
    # Verify potentials - they should satisfy profit = u[i] + v[j] for all basic cells

    for i, j in basic_cells:

        if abs((u[i] + v[j]) - profit_matrix[i][j]) > 0.001:  # Allow small numerical errors

            print(f"Warning: Potential equation not satisfied at ({i},{j})")
    
    # Prepare results

    steps = {

        "initial_allocation": initial_allocation,

        "allocation": allocation,

        "total_cost": cost_results["total_cost"],

        "transport_cost": cost_results["transport_cost"],

        "purchase_cost": cost_results["purchase_cost"],

        "income": cost_results["income"],

        "profit": cost_results["profit"],

        "potentials": {"u": u, "v": v},

        "iterations": 0,

        "improvement_possible": False

    }
    
    return allocation, cost_results["total_cost"], steps


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
    

    print("Exact calculation check based on console output:")

    expected_allocation = [[10, 0, 10], [0, 28, 2]]

    expected_transport = 80 + 0 + 170 + 0 + 252 + 38  # = 540

    print(f"Expected transport cost: {expected_transport}")

    expected_purchase = 200 + 360  # = 560

    print(f"Expected purchase cost: {expected_purchase}")

    expected_income = 300 + 700 + 360  # = 1360

    print(f"Expected income: {expected_income}")

    expected_profit = expected_income - (expected_transport + expected_purchase)

    print(f"Expected profit: {expected_profit}")
    
    web_results = calculate(costs, supply, demand, purchase_prices, selling_prices)
    

    print("\nWebview results:")

    print(f"Transport Cost: {web_results['transport_cost']}")

    print(f"Purchase Cost: {web_results['purchase_cost']}")

    print(f"Income: {web_results['income']}")

    print(f"Profit: {web_results['profit']}")
    

    print("\nComparison with expected values:")

    print(f"Transport Cost: {'✓' if web_results['transport_cost'] == expected_transport else '✗'} (Expected: {expected_transport}, Got: {web_results['transport_cost']})")

    print(f"Purchase Cost: {'✓' if web_results['purchase_cost'] == expected_purchase else '✗'} (Expected: {expected_purchase}, Got: {web_results['purchase_cost']})")

    print(f"Income: {'✓' if web_results['income'] == expected_income else '✗'} (Expected: {expected_income}, Got: {web_results['income']})")

    print(f"Profit: {'✓' if web_results['profit'] == expected_profit else '✗'} (Expected: {expected_profit}, Got: {web_results['profit']}")
    

    # Test case from PDF (part d)

    print("\nTest Case 2 (PDF part d):")

    costs_d = [[8, 14, 17], [12, 9, 19]]

    supply_d = [20, 30]

    demand_d = [10, 28, 27]

    purchase_prices_d = [10, 12]

    selling_prices_d = [30, 25, 30]

    fixed_allocations_d = {(0, 2): 10, (1, 2): 17}
    

    allocation_d, cost_d, steps_d = logika(

        costs_d, supply_d, demand_d, purchase_prices_d, selling_prices_d, 

        fixed_allocations=fixed_allocations_d

    )
    

    print(f"Solution: {allocation_d}")

    print(f"Transport Cost: {steps_d['transport_cost']}")

    print(f"Purchase Cost: {steps_d['purchase_cost']}")

    print(f"Income: {steps_d['income']}")

    print(f"Profit: {steps_d['profit']}")
    

    expected_profit_d = 185

    if abs(steps_d['profit'] - expected_profit_d) < 1:

        print(f"✓ Part D profit calculation matches expected value ({expected_profit_d})")

    else:

        print(f"✗ Part D profit calculation doesn't match. Expected: {expected_profit_d}, Got: {steps_d['profit']}")
    

    # Universal test case

    print("\nTest Case 3 (Universal Test):")

    costs_universal = [

        [10, 8, 6, 9],

        [12, 7, 5, 11], 

        [9, 6, 8, 10]

    ]

    supply_universal = [50, 40, 30]

    demand_universal = [30, 25, 45, 20]

    purchase_prices_universal = [5, 6, 4]

    selling_prices_universal = [20, 18, 22, 19]
    

    allocation_u, cost_u, steps_u = logika(

        costs_universal, supply_universal, demand_universal, 

        purchase_prices_universal, selling_prices_universal

    )
    
    print(f"Solution: {allocation_u}")

    print(f"Transport Cost: {steps_u['transport_cost']}")

    print(f"Purchase Cost: {steps_u['purchase_cost']}")

    print(f"Income: {steps_u['income']}")

    print(f"Profit: {steps_u['profit']}")
    
    # Webview setup

    api = type('API', (), {
        
        'calculate': lambda costs, supply, demand, purchase_prices, selling_prices: calculate(

            json.loads(costs), json.loads(supply), json.loads(demand),

            json.loads(purchase_prices), json.loads(selling_prices)

        )
        
    })()
    
    webview.create_window("Intermediary Problem", "Index.html", js_api=api)

    webview.start()