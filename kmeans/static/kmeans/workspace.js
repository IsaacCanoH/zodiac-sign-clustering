(() => {
    "use strict";

    // ── Tab auto-activation on hash change ──────────────────────────────────
    const activeHash = window.location.hash;
    const urlParams = new URLSearchParams(window.location.search);
    const resultsView = urlParams.get("results_view");

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

    // When the results pane has both algorithms (Bootstrap pills), activate the
    // correct pill based on the `results_view` query parameter.
    if (activeHash === "#results-pane" && resultsView && window.bootstrap) {
        const resultPillByAlgorithm = {
            kmeans: "kmeans-results-tab",
            hierarchical: "hierarchical-results-tab",
        };
        const pillId = resultPillByAlgorithm[resultsView];
        const pill = document.getElementById(pillId);
        if (pill) {
            // Wait for Bootstrap tab to finish showing the pane before switching pill
            const resultTab = document.getElementById("results-tab");
            if (resultTab) {
                resultTab.addEventListener(
                    "shown.bs.tab",
                    () => bootstrap.Tab.getOrCreateInstance(pill).show(),
                    { once: true },
                );
            } else {
                bootstrap.Tab.getOrCreateInstance(pill).show();
            }
        }
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

    // ── Hierarchical result chart ───────────────────────────────────────────
    const hierarchicalChartElement = document.getElementById("hierarchicalClusterChart");
    const hierarchicalChartDataElement = document.getElementById("hierarchical-chart-data");
    if (hierarchicalChartElement && hierarchicalChartDataElement && window.Chart) {
        const chartData = JSON.parse(hierarchicalChartDataElement.textContent);
        const colors = [
            "#0057B8",
            "#E66100",
            "#009E73",
            "#6A00A8",
            "#D7191C",
            "#8C564B",
            "#00A6D6",
            "#B28A00",
            "#CC79A7",
        ];
        let colorIndex = 0;
        const datasets = chartData.clusters.map((cluster) => {
            const color = colors[colorIndex++ % colors.length];
            return {
                label: `Cluster ${cluster.cluster}`,
                data: cluster.points,
                backgroundColor: color,
                borderColor: color,
                pointStyle: "circle",
                pointRadius: 4,
                pointHoverRadius: 6,
            };
        });
        new Chart(hierarchicalChartElement, {
            type: "scatter",
            data: {datasets},
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {mode: "nearest", intersect: true},
                plugins: {
                    legend: {
                        position: "bottom",
                        labels: {usePointStyle: true, boxWidth: 10},
                    },
                    tooltip: {
                        callbacks: {
                            title(items) {
                                return items[0]?.raw
                                    ? `Registro ${items[0].raw.row}`
                                    : "";
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
                                return coordinates;
                            },
                        },
                    },
                },
                scales: {
                    x: {title: {display: true, text: chartData.x_label}},
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
        kmeansForm.addEventListener("submit", () => {
            if (trainButton) {
                trainButton.disabled = true;
                trainButton.textContent = "Entrenando…";
            }
        });
        updateComparisonColumn();
    }

    // ── Hierarchical form logic ──────────────────────────────────────────────
    const hierarchicalForm = document.getElementById("hierarchicalTrainingForm");
    if (hierarchicalForm) {
        const checkboxes = [...hierarchicalForm.querySelectorAll(".hierarchical-column")];
        const selectedCount = document.getElementById("hierarchicalSelectedCount");
        const trainButton = document.getElementById("trainHierarchicalButton");
        const comparisonColumn = document.getElementById("hierarchicalComparisonColumn");
        const toggleColumnsButton = document.getElementById("toggleHierarchicalColumns");

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
        hierarchicalForm.addEventListener("submit", () => {
            if (trainButton) {
                trainButton.disabled = true;
                trainButton.textContent = "Entrenando…";
            }
        });
        updateComparisonColumn();
    }
})();
