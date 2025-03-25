import webview

class App:
    def __init__(self):
        self.window = None

    def start(self):
        self.window = webview.create_window(
            "Metoda Pośrednika - Logistyka",
            "index.html",
            width=1600,  # Ustawienie szerokości na 1600 pikseli
            height=900,  # Ustawienie wysokości na 900 pikseli
            resizable=True
        )
        webview.start()

if __name__ == "__main__":
    app = App()
    app.start()