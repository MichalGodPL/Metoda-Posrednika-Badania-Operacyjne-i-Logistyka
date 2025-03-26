import webview
from Logika import logika

class App:
    def __init__(self):
        self.window = None

    def start(self):
        self.window = webview.create_window(
            "Metoda Pośrednika - Logistyka",
            "index.html",
            width=1600,  # Ustawienie szerokości na 1600 pikseli
            height=900,  # Ustawienie wysokości na 900 pikseli
            resizable=True,
            js_api=API()  # Moved this here
        )
        webview.start()  # Removed api=API()

class API:
    def calculate(self, costs, supply, demand):
        allocation, total_cost = logika(costs, supply, demand)
        return {
            "allocation": allocation,
            "total_cost": total_cost
        }

if __name__ == "__main__":
    app = App()
    app.start()