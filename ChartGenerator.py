import matplotlib.pyplot as plt
import io
import base64

def generate_chart(solution_matrix):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Przykładowa wizualizacja macierzy rozwiązania
    im = ax.imshow(solution_matrix, cmap='Blues')
    
    # Dodanie etykiet
    ax.set_title("Rozwiązanie Metody Pośrednika")
    ax.set_xlabel("Magazyny")
    ax.set_ylabel("Punkty odbioru")
    
    # Zapisz wykres do pamięci jako base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()
    
    return f"data:image/png;base64,{img_base64}"

# Przykład użycia
if __name__ == "__main__":
    sample_matrix = [
        [10, 0, 20],
        [0, 15, 5],
        [30, 0, 10]
    ]
    chart = generate_chart(sample_matrix)
    print("Wykres wygenerowany (base64 string)")