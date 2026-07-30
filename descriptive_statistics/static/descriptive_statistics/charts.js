(() => {
    "use strict";

    if (window.location.hash === "#statistics-pane" && window.bootstrap) {
        const statisticsTab = document.getElementById("statistics-tab");
        if (statisticsTab) {
            bootstrap.Tab.getOrCreateInstance(statisticsTab).show();
            document
                .getElementById("statistics-pane")
                ?.scrollIntoView({block: "start"});
        }
    }

    const dataElement = document.getElementById("statistics-chart-data");
    if (!dataElement || typeof Chart === "undefined") return;

    const data = JSON.parse(dataElement.textContent);
    const analysis = data.analysis;
    const primaryCanvas = document.getElementById("primaryStatisticsChart");
    const secondaryCanvas = document.getElementById("secondaryStatisticsChart");

    const commonOptions = {
        responsive: true,
        plugins: {
            legend: {display: false},
        },
        scales: {
            y: {
                beginAtZero: true,
                ticks: {precision: 0},
                title: {display: true, text: "Frecuencia"},
            },
        },
    };

    if (analysis && primaryCanvas && secondaryCanvas) {
        const chart = analysis.chart;
        const hasLongCategories =
            analysis.kind === "qualitative" &&
            chart.labels.some((label) => String(label).length > 18);
        const primaryOptions = hasLongCategories
            ? {
                responsive: true,
                plugins: {
                    legend: {display: false},
                    tooltip: {
                        callbacks: {
                            title: (items) =>
                                chart.labels[items[0].dataIndex],
                            label: (item) => `Frecuencia: ${item.raw}`,
                        },
                    },
                },
                scales: {
                    x: {
                        ticks: {display: false},
                        grid: {display: false},
                    },
                    y: {
                        beginAtZero: true,
                        ticks: {precision: 0},
                        title: {display: true, text: "Frecuencia"},
                    },
                },
            }
            : commonOptions;

        new Chart(primaryCanvas, {
            type: "bar",
            data: {
                labels: chart.labels,
                datasets: [{
                    label: "Frecuencia",
                    data: chart.frequencies,
                    backgroundColor: "rgba(13, 110, 253, 0.65)",
                    borderColor: "rgb(13, 110, 253)",
                    borderWidth: 1,
                }],
            },
            options: primaryOptions,
        });

        if (analysis.kind === "quantitative") {
            new Chart(secondaryCanvas, {
                type: "line",
                data: {
                    labels: chart.labels,
                    datasets: [{
                        label: "Frecuencia",
                        data: chart.frequencies,
                        borderColor: "rgb(13, 110, 253)",
                        backgroundColor: "rgb(13, 110, 253)",
                        pointRadius: 4,
                        tension: 0,
                    }],
                },
                options: commonOptions,
            });
        } else {
            new Chart(secondaryCanvas, {
                type: "pie",
                data: {
                    labels: chart.labels,
                    datasets: [{
                        data: chart.frequencies,
                        backgroundColor: [
                            "#0d6efd", "#6f42c1", "#d63384", "#dc3545",
                            "#fd7e14", "#ffc107", "#198754", "#20c997",
                            "#0dcaf0", "#6c757d",
                        ],
                    }],
                },
                options: {
                    responsive: true,
                    plugins: {legend: {position: "bottom"}},
                },
            });
        }
    }

    const scatter = data.scatter;
    const scatterCanvas = document.getElementById("scatterStatisticsChart");
    if (scatter && scatterCanvas) {
        new Chart(scatterCanvas, {
            type: "scatter",
            data: {
                datasets: [{
                    label: `${scatter.first_column} / ${scatter.second_column}`,
                    data: scatter.points,
                    backgroundColor: "rgba(13, 110, 253, 0.65)",
                    pointRadius: 4,
                }],
            },
            options: {
                responsive: true,
                plugins: {legend: {display: false}},
                scales: {
                    x: {title: {display: true, text: scatter.first_column}},
                    y: {title: {display: true, text: scatter.second_column}},
                },
            },
        });
    }
})();
