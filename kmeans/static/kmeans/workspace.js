(() => {
    "use strict";

    const activeHash = window.location.hash;
    if (
        ["#training-pane", "#results-pane"].includes(activeHash) &&
        window.bootstrap
    ) {
        const tabId =
            activeHash === "#training-pane" ? "training-tab" : "results-tab";
        const tab = document.getElementById(tabId);
        if (tab) bootstrap.Tab.getOrCreateInstance(tab).show();
    }

    const form = document.getElementById("kmeansTrainingForm");
    if (!form) return;

    const checkboxes = [...form.querySelectorAll(".kmeans-column")];
    const selectedCount = document.getElementById("kmeansSelectedCount");
    const trainButton = document.getElementById("trainKMeansButton");
    const comparisonColumn = document.getElementById("comparisonColumn");

    const updateCount = () => {
        const count = checkboxes.filter((checkbox) => checkbox.checked).length;
        selectedCount.textContent =
            `${count} ${count === 1 ? "columna seleccionada" : "columnas seleccionadas"}`;
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

    document.getElementById("selectAllKMeansColumns")?.addEventListener(
        "click",
        () => {
            checkboxes.forEach((checkbox) => {
                if (!checkbox.disabled) checkbox.checked = true;
            });
            updateCount();
        },
    );
    document.getElementById("clearKMeansColumns")?.addEventListener(
        "click",
        () => {
            checkboxes.forEach((checkbox) => {
                checkbox.checked = false;
            });
            updateCount();
        },
    );
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
