document.addEventListener('DOMContentLoaded', () => {
    let heatmapInstance = null;
    let networkInstance = null;

    const heatmapCtx = document.getElementById('heatmapChart').getContext('2d');
    const networkCtx = document.getElementById('networkChart').getContext('2d');

    Alpine.effect(() => {
        const root = document.querySelector('[x-data]');
        if (!root || !root.__x) {
            console.log('Brak globalnego obiektu Alpine.');
            return;
        }
        const xData = root.__x.$data;
        console.log('Stan xData:', xData);

        if (xData.tab === 'chart' && xData.resultData) {
            console.log('Dane resultData:', xData.resultData); // Logowanie danych
            const data = xData.resultData;
            const costs = xData.costs;
            const allocation = data.allocation;

            // Zniszcz poprzednie wykresy, jeśli istnieją
            if (heatmapInstance) heatmapInstance.destroy();
            if (networkInstance) networkInstance.destroy();

            // 1. Macierz ciepła (Heatmap)
            const heatmapData = {
                labels: {
                    x: allocation[0].map((_, j) => `O${j + 1}`),
                    y: allocation.map((_, i) => `D${i + 1}`)
                },
                datasets: [{
                    label: 'Alokacja',
                    data: allocation.flatMap((row, i) =>
                        row.map((value, j) => ({
                            x: `O${j + 1}`,
                            y: `D${i + 1}`,
                            v: value,
                            cost: costs[i][j]
                        }))
                    ),
                    backgroundColor: (ctx) => {
                        const value = ctx.raw.v;
                        const maxValue = Math.max(...allocation.flat());
                        const intensity = value / maxValue;
                        return `rgba(75, 192, 192, ${intensity})`;
                    },
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1,
                    width: ({ chart }) => (chart.chartArea || {}).width / allocation[0].length - 1,
                    height: ({ chart }) => (chart.chartArea || {}).height / allocation.length - 1
                }]
            };

            heatmapInstance = new Chart(heatmapCtx, {
                type: 'matrix',
                data: heatmapData,
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            title: { display: true, text: 'Odbiorcy', color: '#ffffff' },
                            ticks: { color: '#ffffff' }
                        },
                        y: {
                            title: { display: true, text: 'Dostawcy', color: '#ffffff' },
                            ticks: { color: '#ffffff' }
                        }
                    },
                    plugins: {
                        legend: { labels: { color: '#ffffff' } },
                        tooltip: {
                            callbacks: {
                                label: (context) => {
                                    const v = context.raw.v;
                                    const cost = context.raw.cost;
                                    return `Alokacja: ${v}, Koszt jednostkowy: ${cost}`;
                                }
                            }
                        }
                    }
                }
            });

            // 2. Graf sieciowy (Network Graph)
            const nodes = [];
            const links = [];
            const nodeMap = new Map();

            allocation.forEach((_, i) => {
                nodes.push({ id: `D${i + 1}`, group: 'dostawca' });
                nodeMap.set(`D${i + 1}`, nodes.length - 1);
            });

            allocation[0].forEach((_, j) => {
                nodes.push({ id: `O${j + 1}`, group: 'odbiorca' });
                nodeMap.set(`O${j + 1}`, nodes.length - 1);
            });

            allocation.forEach((row, i) => {
                row.forEach((value, j) => {
                    if (value > 0) {
                        links.push({
                            source: `D${i + 1}`,
                            target: `O${j + 1}`,
                            value: value,
                            cost: costs[i][j]
                        });
                    }
                });
            });

            const networkData = {
                datasets: [
                    {
                        label: 'Węzły',
                        data: nodes.map((node, idx) => ({
                            x: Math.cos((idx / nodes.length) * 2 * Math.PI) * 10,
                            y: Math.sin((idx / nodes.length) * 2 * Math.PI) * 10,
                            id: node.id,
                            group: node.group
                        })),
                        backgroundColor: (ctx) => (ctx.raw.group === 'dostawca' ? 'rgba(255, 99, 132, 0.8)' : 'rgba(54, 162, 235, 0.8)'),
                        pointRadius: 10,
                        pointHoverRadius: 15
                    },
                    {
                        label: 'Połączenia',
                        data: links.flatMap(link => [
                            {
                                x: nodes[nodeMap.get(link.source)].x || Math.cos((nodeMap.get(link.source) / nodes.length) * 2 * Math.PI) * 10,
                                y: nodes[nodeMap.get(link.source)].y || Math.sin((nodeMap.get(link.source) / nodes.length) * 2 * Math.PI) * 10
                            },
                            {
                                x: nodes[nodeMap.get(link.target)].x || Math.cos((nodeMap.get(link.target) / nodes.length) * 2 * Math.PI) * 10,
                                y: nodes[nodeMap.get(link.target)].y || Math.sin((nodeMap.get(link.target) / nodes.length) * 2 * Math.PI) * 10
                            }
                        ]),
                        type: 'line',
                        borderColor: 'rgba(75, 192, 192, 0.5)',
                        borderWidth: (ctx) => links[Math.floor(ctx.dataIndex / 2)].value / Math.max(...links.map(l => l.value)) * 5,
                        pointRadius: 0
                    }
                ]
            };

            networkInstance = new Chart(networkCtx, {
                type: 'scatter',
                data: networkData,
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { display: false },
                        y: { display: false }
                    },
                    plugins: {
                        legend: { labels: { color: '#ffffff' } },
                        tooltip: {
                            callbacks: {
                                label: (context) => {
                                    if (context.dataset.label === 'Węzły') {
                                        return `${context.raw.id} (${context.raw.group})`;
                                    } else {
                                        const link = links[Math.floor(context.dataIndex / 2)];
                                        return `Przepływ: ${link.source} -> ${link.target}, Wartość: ${link.value}, Koszt: ${link.cost}`;
                                    }
                                }
                            }
                        }
                    }
                }
            });
        } else {
            console.log('Brak danych do wyświetlenia wykresów:', { tab: xData.tab, resultData: xData.resultData });
        }
    });
});