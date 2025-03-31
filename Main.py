import webview

from Logika import logika

from Charts import generate_chart_images

import json


class App:
    def __init__(self):
        self.window = None

    def start(self):
        self.window = webview.create_window(
            "Metoda Pośrednika - Logistyka",
            "index.html",
            width=1600,
            height=900,
            resizable=True,
            js_api=API()
        )
        webview.start()

class API:
    def calculate(self, costs, supply, demand, purchase_prices, selling_prices):
        # Parsowanie danych z JSON, bo frontend przesyła je jako stringi
        costs = json.loads(costs)
        supply = json.loads(supply)
        demand = json.loads(demand)
        purchase_prices = json.loads(purchase_prices)
        selling_prices = json.loads(selling_prices)
        
        # Wywołanie logiki obliczeń z Logika.py
        allocation, total_cost, steps = logika(costs, supply, demand, purchase_prices, selling_prices)
        
        # Zwrócenie wyników obliczeń
        return {
            "allocation": steps["allocation"],
            "total_cost": total_cost,
            "transport_cost": steps["transport_cost"],
            "purchase_cost": steps["purchase_cost"],
            "income": steps["income"],
            "profit": steps["profit"],
            "initial_allocation": steps["initial_allocation"],
            "potentials": steps["potentials"],
            "iterations": steps["iterations"],
            "improvement_possible": steps["improvement_possible"]
        }

    def generate_chart_images(self, costs, steps):
        # Parsowanie danych z JSON
        costs = json.loads(costs)
        steps = json.loads(steps)
        
        # Wywołanie generowania wykresów z Charts.py
        chart_data = generate_chart_images(costs, steps)
        
        # Zwrócenie danych obrazów w formacie base64
        return {
            "heatmap_image": chart_data["heatmap_image"],
            "network_image": chart_data["network_image"]
        }

if __name__ == "__main__":
    app = App()
    app.start()