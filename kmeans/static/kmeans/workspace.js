(() => {
    "use strict";

    // ── Tab auto-activation on hash change ──────────────────────────────────
    const activeHash = window.location.hash;
    const urlParams = new URLSearchParams(window.location.search);
    const resultsView = urlParams.get("results_view"); // 'kmeans' | 'dbscan' | null

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
        const pillId =
            resultsView === "dbscan" ? "dbscan-results-tab" : "kmeans-results-tab";
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
        datasets.push({
            label: "Centroides",
            data: chartData.centroids.map((centroid) => ({
                ...centroid,
                isCentroid: true,
            })),
            backgroundColor: "#212529",
            borderColor: "#ffffff",
            borderWidth: 2,
            pointStyle: "crossRot",
            pointRadius: 9,
            pointHoverRadius: 11,
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

    // DBSCAN result chart
    const dbscanChartElement = document.getElementById("dbscanClusterChart");
    const dbscanChartDataElement = document.getElementById("dbscan-chart-data");
    if (dbscanChartElement && dbscanChartDataElement && window.Chart) {
        const chartData = JSON.parse(dbscanChartDataElement.textContent);
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
        const datasets = chartData.groups.map((group) => {
            const isNoise = group.cluster === -1;
            const color = isNoise ? "#6C757D" : colors[colorIndex++ % colors.length];
            return {
                label: group.label,
                data: group.points,
                backgroundColor: color,
                borderColor: color,
                pointStyle: "circle",
                pointRadius: isNoise ? 3 : 4,
                pointHoverRadius: 6,
            };
        });
        new Chart(dbscanChartElement, {
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

    // ── Algorithm selector (dropdown in training pane) ───────────────────────
    const STORAGE_KEY = "zodiac_active_algorithm";
    const kmeansPanelEl = document.getElementById("kmeans-form-panel");
    const dbscanPanelEl = document.getElementById("dbscan-form-panel");
    const algorithmLabelEl = document.getElementById("algorithmLabel");
    const dropdownItems = document.querySelectorAll("[data-algorithm]");

    /**
     * Show the form panel for `algorithm` and hide the other.
     * Also marks the matching dropdown item as active.
     */
    function activateAlgorithm(algorithm) {
        if (!kmeansPanelEl || !dbscanPanelEl) return;

        if (algorithm === "dbscan") {
            kmeansPanelEl.style.display = "none";
            dbscanPanelEl.style.display = "";
            if (algorithmLabelEl) algorithmLabelEl.textContent = "DBSCAN";
        } else {
            dbscanPanelEl.style.display = "none";
            kmeansPanelEl.style.display = "";
            if (algorithmLabelEl) algorithmLabelEl.textContent = "K-Means";
        }

        // Update active state on dropdown items
        dropdownItems.forEach((item) => {
            item.classList.toggle("active", item.dataset.algorithm === algorithm);
        });

        try {
            sessionStorage.setItem(STORAGE_KEY, algorithm);
        } catch (_) {
            // sessionStorage unavailable — ignore
        }
    }

    // Attach click handlers to dropdown items
    dropdownItems.forEach((item) => {
        item.addEventListener("click", () => {
            activateAlgorithm(item.dataset.algorithm);
        });
    });

    // Determine initial algorithm:
    //   1. Server-side hint (from session after a form error / redirect)
    //   2. sessionStorage (last user choice)
    //   3. Default: kmeans
    const serverAlgorithm =
        typeof window.__activeAlgorithm !== "undefined"
            ? window.__activeAlgorithm
            : null;
    let storedAlgorithm = null;
    try {
        storedAlgorithm = sessionStorage.getItem(STORAGE_KEY);
    } catch (_) {}

    const initialAlgorithm = serverAlgorithm || storedAlgorithm || "kmeans";
    activateAlgorithm(initialAlgorithm);

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

    // ── DBSCAN form logic ────────────────────────────────────────────────────
    const dbscanForm = document.getElementById("dbscanTrainingForm");
    if (dbscanForm) {
        const checkboxes = [...dbscanForm.querySelectorAll(".dbscan-column")];
        const selectedCount = document.getElementById("dbscanSelectedCount");
        const trainButton = document.getElementById("trainDbscanButton");
        const comparisonColumn = document.getElementById("dbscanComparisonColumn");
        const toggleColumnsButton = document.getElementById("toggleDbscanColumns");

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
        dbscanForm.addEventListener("submit", () => {
            if (trainButton) {
                trainButton.disabled = true;
                trainButton.textContent = "Entrenando…";
            }
        });
        updateComparisonColumn();
    }
})();
