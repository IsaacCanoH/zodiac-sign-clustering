(() => {
    "use strict";

    const activeHash = window.location.hash;
    const hashTabs = {
        "#training-pane": "training-tab",
        "#results-pane": "results-tab",
    };
    if (hashTabs[activeHash] && window.bootstrap) {
        const tabId = hashTabs[activeHash];
        const tab = document.getElementById(tabId);
        if (tab) bootstrap.Tab.getOrCreateInstance(tab).show();
    }

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

    const form = document.getElementById("kmeansTrainingForm");
    if (!form) return;

    const checkboxes = [...form.querySelectorAll(".kmeans-column")];
    const selectedCount = document.getElementById("kmeansSelectedCount");
    const trainButton = document.getElementById("trainKMeansButton");
    const comparisonColumn = document.getElementById("comparisonColumn");
    const toggleColumnsButton = document.getElementById("toggleKMeansColumns");

    const updateCount = () => {
        const count = checkboxes.filter((checkbox) => checkbox.checked).length;
        selectedCount.textContent =
            `${count} ${count === 1 ? "columna seleccionada" : "columnas seleccionadas"}`;
        const availableCheckboxes = checkboxes.filter(
            (checkbox) => !checkbox.disabled,
        );
        const allSelected =
            availableCheckboxes.length > 0 &&
            availableCheckboxes.every((checkbox) => checkbox.checked);
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
        const availableCheckboxes = checkboxes.filter(
            (checkbox) => !checkbox.disabled,
        );
        const shouldSelect = !availableCheckboxes.every(
            (checkbox) => checkbox.checked,
        );
        availableCheckboxes.forEach((checkbox) => {
            checkbox.checked = shouldSelect;
        });
        updateCount();
    });
    checkboxes.forEach((checkbox) => {
        checkbox.addEventListener("change", updateCount);
    });
    comparisonColumn?.addEventListener("change", updateComparisonColumn);
    form.addEventListener("submit", () => {
        trainButton.disabled = true;
        trainButton.textContent = "Entrenando…";
    });
    updateComparisonColumn();
})();
