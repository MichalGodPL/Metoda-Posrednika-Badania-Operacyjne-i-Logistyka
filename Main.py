import webview

class App:
    def __init__(self):
        self.window = None

    def start(self):
        self.window = webview.create_window(
            "Metoda Pośrednika - Logistyka",
            "index.html",
            width=1200,
            height=800,
            resizable=True
        )
        webview.start()

if __name__ == "__main__":
    app = App()
    app.start()