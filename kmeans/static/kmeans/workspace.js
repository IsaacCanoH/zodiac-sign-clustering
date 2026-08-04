(() => {
    "use strict";

    // ── Tab auto-activation on hash change ──────────────────────────────────
    const activeHash = window.location.hash;

    const dashboardTabByHash = {
        "#data-pane": "data-tab",
        "#statistics-pane": "statistics-tab",
        "#training-pane": "training-tab",
        "#results-pane": "results-tab",
        "#classified-pane": "classified-tab",
        "#models-pane": "models-tab",
    };
    if (dashboardTabByHash[activeHash] && window.bootstrap) {
        const tabId = dashboardTabByHash[activeHash];
        const tab = document.getElementById(tabId);
        if (tab) bootstrap.Tab.getOrCreateInstance(tab).show();
    }

    // ── K-Means result chart ────────────────────────────────────────────────
    const chartElement = document.getElementById("kmeansClusterChart");
    const chartDataElement = document.getElementById("kmeans-chart-data");
    if (chartElement && chartDataElement && window.Chart) {
        const chartData = JSON.parse(chartDataElement.textContent);
        const colors = [
            "#0057B8",
            "#E66100",
            "#009E73",
            "#6A00A8",
            "#D7191C",
            "#8C564B",
            "#00A6D6",
            "#B28A00",
            "#444444",
            "#CC79A7",
        ];
        const datasets = chartData.clusters.map((cluster, index) => ({
            label: `Cluster ${cluster.cluster}`,
            data: cluster.points,
            backgroundColor: colors[index % colors.length],
            borderColor: colors[index % colors.length],
            pointRadius: 4,
            pointHoverRadius: 6,
        }));
        chartData.centroids.forEach((centroid) => {
            const clusterIndex = chartData.clusters.findIndex(
                (cluster) => cluster.cluster === centroid.cluster,
            );
            const centroidColor = colors[
                (clusterIndex >= 0 ? clusterIndex : centroid.cluster - 1) % colors.length
            ];
            datasets.push({
                label: `Centroide Cluster ${centroid.cluster}`,
                data: [{...centroid, isCentroid: true}],
                backgroundColor: centroidColor,
                borderColor: "#212529",
                borderWidth: 2,
                pointStyle: "circle",
                pointRadius: 6,
                pointHoverRadius: 8,
            });
        });
        new Chart(chartElement, {
            type: "scatter",
            data: {datasets},
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {mode: "nearest", intersect: true},
                plugins: {
                    legend: {
                        position: "bottom",
                        // La leyenda conserva un tamaño uniforme aunque los
                        // centroides sean ligeramente mayores en la gráfica.
                        labels: {usePointStyle: true, boxWidth: 10},
                    },
                    tooltip: {
                        callbacks: {
                            title(items) {
                                const point = items[0]?.raw;
                                if (!point) return "";
                                return point.isCentroid
                                    ? `Centroide del cluster ${point.cluster}`
                                    : `Registro ${point.row}`;
                            },
                            label(context) {
                                const point = context.raw;
                                const coordinates = [
                                    `${chartData.x_label}: ${Number(point.x).toFixed(3)}`,
                                ];
                                if (chartData.y_label) {
                                    coordinates.push(
                                        `${chartData.y_label}: ${Number(point.y).toFixed(3)}`,
                                    );
                                }
                                if (!point.isCentroid) {
                                    coordinates.push(
                                        `Cluster asignado: ${point.cluster}`,
                                        `Distancia al centroide ${point.cluster}: ${Number(point.centroid_distance).toFixed(3)}`,
                                        `Segundo más cercano: Cluster ${point.second_cluster} — ${Number(point.second_centroid_distance).toFixed(3)}`,
                                        `Diferencia: ${Number(point.distance_difference).toFixed(3)}`,
                                        `Evaluación: ${point.assignment_confidence}`,
                                    );
                                }
                                return coordinates;
                            },
                        },
                    },
                },
                scales: {
                    x: {
                        title: {display: true, text: chartData.x_label},
                    },
                    y: {
                        display: Boolean(chartData.y_label),
                        title: {
                            display: Boolean(chartData.y_label),
                            text: chartData.y_label,
                        },
                        suggestedMin: chartData.y_label ? undefined : -1,
                        suggestedMax: chartData.y_label ? undefined : 1,
                    },
                },
            },
        });
    }

    const selectionCanvas = document.getElementById("kmeansSelectionChart");
    const selectionDataElement = document.getElementById("kmeans-analysis-data");
    if (selectionCanvas && selectionDataElement && window.Chart) {
        const analysis = JSON.parse(selectionDataElement.textContent);
        const candidates = analysis.results;
        new Chart(selectionCanvas, {
            type: "line",
            data: {
                labels: candidates.map((item) => `k=${item.k}`),
                datasets: [
                    {
                        label: "Inercia (menor es mejor)",
                        data: candidates.map((item) => item.inertia),
                        borderColor: "#6c757d",
                        backgroundColor: "#6c757d",
                        yAxisID: "inertia",
                        tension: 0.2,
                    },
                    {
                        label: "Silueta (mayor es mejor)",
                        data: candidates.map((item) => item.silhouette),
                        borderColor: "#0057B8",
                        backgroundColor: "#0057B8",
                        yAxisID: "score",
                        tension: 0.2,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {mode: "index", intersect: false},
                plugins: {legend: {position: "bottom"}},
                scales: {
                    inertia: {type: "linear", position: "left", title: {display: true, text: "Inercia"}},
                    score: {type: "linear", position: "right", min: -1, max: 1, grid: {drawOnChartArea: false}, title: {display: true, text: "Silueta"}},
                },
            },
        });
    }

    // ── K-Means form logic ───────────────────────────────────────────────────
    const kmeansForm = document.getElementById("kmeansTrainingForm");
    if (kmeansForm) {
        const checkboxes = [...kmeansForm.querySelectorAll(".kmeans-column")];
        const selectedCount = document.getElementById("kmeansSelectedCount");
        const trainButton = document.getElementById("trainKMeansButton");
        const comparisonColumn = document.getElementById("comparisonColumn");
        const toggleColumnsButton = document.getElementById("toggleKMeansColumns");

        const updateCount = () => {
            const count = checkboxes.filter((cb) => cb.checked).length;
            if (selectedCount) {
                selectedCount.textContent =
                    `${count} ${count === 1 ? "columna seleccionada" : "columnas seleccionadas"}`;
            }
            const availableCheckboxes = checkboxes.filter((cb) => !cb.disabled);
            const allSelected =
                availableCheckboxes.length > 0 &&
                availableCheckboxes.every((cb) => cb.checked);
            if (toggleColumnsButton) {
                toggleColumnsButton.textContent = allSelected
                    ? "Deseleccionar todas"
                    : "Seleccionar todas";
                toggleColumnsButton.disabled = availableCheckboxes.length === 0;
            }
        };

        const updateComparisonColumn = () => {
            const selectedComparison = comparisonColumn?.value || "";
            checkboxes.forEach((checkbox) => {
                const isComparison = checkbox.value === selectedComparison;
                checkbox.disabled = isComparison;
                if (isComparison) checkbox.checked = false;
                checkbox.closest(".form-check")?.classList.toggle(
                    "bg-body-tertiary",
                    isComparison,
                );
            });
            updateCount();
        };

        toggleColumnsButton?.addEventListener("click", () => {
            const availableCheckboxes = checkboxes.filter((cb) => !cb.disabled);
            const shouldSelect = !availableCheckboxes.every((cb) => cb.checked);
            availableCheckboxes.forEach((cb) => {
                cb.checked = shouldSelect;
            });
            updateCount();
        });
        checkboxes.forEach((cb) => cb.addEventListener("change", updateCount));
        comparisonColumn?.addEventListener("change", updateComparisonColumn);
        kmeansForm.addEventListener("submit", (event) => {
            const submitter = event.submitter;
            if (submitter) {
                submitter.disabled = true;
                submitter.textContent = submitter.id === "analyzeKMeansButton"
                    ? "Analizando…"
                    : "Entrenando…";
            }
        });
        updateComparisonColumn();
    }

})();
