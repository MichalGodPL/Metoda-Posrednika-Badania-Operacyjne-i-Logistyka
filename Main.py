import webview
from Logika import logika

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
    def calculate(self, costs, supply, demand):
        allocation, total_cost, steps = logika(costs, supply, demand)
        return {
            "allocation": allocation,
            "total_cost": total_cost,
            "initial_allocation": steps["initial_allocation"],
            "potentials": steps["potentials"]
        }

if __name__ == "__main__":
    app = App()
    app.start()