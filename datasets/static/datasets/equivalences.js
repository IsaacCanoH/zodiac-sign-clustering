(() => {
    "use strict";

    const dataElement = document.getElementById("equivalence-data");
    const form = document.getElementById("equivalenceForm");
    if (!dataElement || !form) return;

    const data = JSON.parse(dataElement.textContent);
    const columns = data.numeric_columns || [];
    const configurations = data.configurations || [];
    const columnByName = new Map(columns.map((column) => [column.name, column]));
    const configurationById = new Map(
        configurations.map((configuration) => [String(configuration.id), configuration])
    );

    const elements = {
        modal: document.getElementById("equivalenceModal"),
        saved: document.getElementById("savedConfiguration"),
        reference: document.getElementById("referenceColumn"),
        referenceType: document.getElementById("referenceType"),
        referenceValid: document.getElementById("referenceValidCount"),
        referenceNull: document.getElementById("referenceNullCount"),
        referenceValues: document.getElementById("referenceValues"),
        rows: document.getElementById("equivalenceRows"),
        name: document.getElementById("configurationName"),
        nameError: document.getElementById("configurationNameError"),
        mappingError: document.getElementById("equivalenceError"),
        columnsError: document.getElementById("columnsError"),
        columnList: document.getElementById("numericColumnList"),
        search: document.getElementById("columnSearch"),
        showAll: document.getElementById("showAllNumericColumns"),
        selectedCount: document.getElementById("selectedColumnCount"),
        toggleCompatible: document.getElementById("toggleCompatibleColumnsButton"),
        alert: document.getElementById("equivalenceFormAlert"),
        compatibilityWarning: document.getElementById("compatibilityWarning"),
        save: document.getElementById("saveEquivalenceButton"),
        removeApplication: document.getElementById("removeApplicationButton"),
        deleteConfiguration: document.getElementById("deleteConfigurationButton"),
    };

    let currentConfigurationId = "";
    let selectedColumns = new Set();

    function canonicalClientNumber(value) {
        const trimmed = String(value).trim();
        if (!trimmed) return "";
        const number = Number(trimmed);
        return Number.isFinite(number) ? String(number) : trimmed;
    }

    function configurationUrl(template, configurationId) {
        return template.replace("/0/", `/${configurationId}/`);
    }

    function csrfToken() {
        return form.querySelector("[name=csrfmiddlewaretoken]").value;
    }

    function showAlert(message, type = "danger") {
        elements.alert.className = `alert alert-${type}`;
        elements.alert.textContent = message;
    }

    function hideAlert() {
        elements.alert.className = "alert d-none";
        elements.alert.textContent = "";
    }

    function setFieldError(element, message) {
        element.textContent = message || "";
        element.classList.toggle("d-none", !message);
    }

    function clearErrors() {
        hideAlert();
        elements.name.classList.remove("is-invalid");
        elements.nameError.textContent = "";
        setFieldError(elements.mappingError, "");
        setFieldError(elements.columnsError, "");
        elements.compatibilityWarning.classList.add("d-none");
        elements.compatibilityWarning.replaceChildren();
    }

    function availableColumns() {
        const recommended = columns.filter((column) => column.recommended);
        return elements.showAll.checked || recommended.length === 0 ? columns : recommended;
    }

    function populateSavedConfigurations() {
        configurations.forEach((configuration) => {
            const option = document.createElement("option");
            option.value = configuration.id;
            option.textContent = `${configuration.name}${configuration.applied ? " — Aplicada" : ""}`;
            elements.saved.append(option);
        });
    }

    function populateReferenceColumns(preferredName = "") {
        const previousValue = preferredName || elements.reference.value;
        elements.reference.replaceChildren();
        availableColumns().forEach((column) => {
            const option = document.createElement("option");
            option.value = column.name;
            option.textContent = `${column.name} — ${column.reason}`;
            elements.reference.append(option);
        });
        if ([...elements.reference.options].some((option) => option.value === previousValue)) {
            elements.reference.value = previousValue;
        }
        renderReferenceSummary();
    }

    function renderReferenceSummary() {
        const column = columnByName.get(elements.reference.value);
        elements.referenceType.textContent = column?.type || "—";
        elements.referenceValid.textContent = column?.valid_count ?? 0;
        elements.referenceNull.textContent = column?.null_count ?? 0;
        elements.referenceValues.textContent = column?.unique_values.join(", ") || "—";
    }

    function createMappingRow(value = "", label = "", manual = true) {
        const row = document.createElement("tr");
        row.dataset.manual = manual ? "true" : "false";

        const valueCell = document.createElement("td");
        const valueInput = document.createElement("input");
        valueInput.className = "form-control form-control-sm";
        valueInput.type = "number";
        valueInput.step = "any";
        valueInput.value = value;
        valueInput.setAttribute("aria-label", "Valor cuantitativo");
        valueCell.append(valueInput);

        const labelCell = document.createElement("td");
        const labelInput = document.createElement("input");
        labelInput.className = "form-control form-control-sm";
        labelInput.value = label;
        labelInput.placeholder = "Escribe el significado";
        labelInput.setAttribute("aria-label", `Significado de ${value || "nuevo valor"}`);
        labelCell.append(labelInput);

        const actionCell = document.createElement("td");
        actionCell.className = "text-end";
        const removeButton = document.createElement("button");
        removeButton.type = "button";
        removeButton.className = "btn btn-outline-danger btn-sm";
        removeButton.textContent = "Eliminar";
        removeButton.disabled = !manual;
        removeButton.title = manual
            ? "Eliminar valor agregado"
            : "Los valores detectados no se pueden eliminar";
        removeButton.addEventListener("click", () => {
            row.remove();
            renderColumnList();
        });
        actionCell.append(removeButton);

        [valueInput, labelInput].forEach((input) => {
            input.addEventListener("input", () => {
                input.classList.remove("is-invalid");
                renderColumnList();
            });
        });
        valueInput.addEventListener("blur", sortMappingRows);

        row.append(valueCell, labelCell, actionCell);
        elements.rows.append(row);
    }

    function mappingRows() {
        return [...elements.rows.querySelectorAll("tr")].map((row) => {
            const inputs = row.querySelectorAll("input");
            return {
                value: inputs[0].value.trim(),
                label: inputs[1].value.trim(),
                row,
            };
        });
    }

    function sortMappingRows() {
        [...elements.rows.children]
            .sort((left, right) => {
                const leftText = left.querySelector("input").value.trim();
                const rightText = right.querySelector("input").value.trim();
                if (!leftText) return 1;
                if (!rightText) return -1;
                const leftValue = Number(leftText);
                const rightValue = Number(rightText);
                if (!Number.isFinite(leftValue)) return 1;
                if (!Number.isFinite(rightValue)) return -1;
                return leftValue - rightValue;
            })
            .forEach((row) => elements.rows.append(row));
    }

    function configuredValues() {
        return new Set(
            mappingRows()
                .map((item) => canonicalClientNumber(item.value))
                .filter(Boolean)
        );
    }

    function resetMappingFromReference() {
        const column = columnByName.get(elements.reference.value);
        elements.rows.replaceChildren();
        (column?.unique_values || []).forEach((value) => {
            createMappingRow(value, "", false);
        });
        renderColumnList();
    }

    function compatibilityFor(column) {
        const configured = configuredValues();
        const missing = column.unique_values.filter((value) => !configured.has(value));
        return {compatible: missing.length === 0, missing};
    }

    function renderColumnList() {
        const search = elements.search.value.trim().toLocaleLowerCase("es");
        elements.columnList.replaceChildren();

        availableColumns()
            .filter((column) => column.name.toLocaleLowerCase("es").includes(search))
            .forEach((column) => {
                const compatibility = compatibilityFor(column);
                const label = document.createElement("label");
                label.className = "list-group-item list-group-item-action d-flex gap-3";

                const checkbox = document.createElement("input");
                checkbox.type = "checkbox";
                checkbox.className = "form-check-input flex-shrink-0";
                checkbox.checked = selectedColumns.has(column.name);
                checkbox.addEventListener("change", () => {
                    if (checkbox.checked) selectedColumns.add(column.name);
                    else selectedColumns.delete(column.name);
                    updateSelectedCount();
                });

                const content = document.createElement("span");
                content.className = "w-100";
                const heading = document.createElement("span");
                heading.className = "d-flex flex-wrap justify-content-between gap-2";
                const name = document.createElement("strong");
                name.textContent = column.name;
                const badge = document.createElement("span");
                badge.className = compatibility.compatible
                    ? "badge text-bg-success"
                    : "badge text-bg-warning";
                badge.textContent = compatibility.compatible
                    ? "Compatible"
                    : `Faltan: ${compatibility.missing.join(", ")}`;
                heading.append(name, badge);
                const details = document.createElement("small");
                details.className = "text-secondary d-block mt-1";
                details.textContent = `Valores: ${column.unique_values.join(", ")} — ${column.reason}`;
                content.append(heading, details);
                label.append(checkbox, content);
                elements.columnList.append(label);
            });

        if (!elements.columnList.children.length) {
            const empty = document.createElement("div");
            empty.className = "p-3 text-secondary text-center";
            empty.textContent = "No se encontraron columnas con ese criterio.";
            elements.columnList.append(empty);
        }
        updateSelectedCount();
    }

    function updateSelectedCount() {
        elements.selectedCount.textContent = selectedColumns.size;
        const compatibleColumns = availableColumns().filter(
            (column) => compatibilityFor(column).compatible
        );
        const allCompatibleSelected =
            compatibleColumns.length > 0 &&
            compatibleColumns.every((column) => selectedColumns.has(column.name));
        if (elements.toggleCompatible) {
            elements.toggleCompatible.textContent = allCompatibleSelected
                ? "Deseleccionar todas"
                : "Seleccionar compatibles";
            elements.toggleCompatible.disabled = compatibleColumns.length === 0;
        }
    }

    function initializeNewConfiguration() {
        currentConfigurationId = "";
        selectedColumns = new Set();
        elements.saved.value = "";
        elements.name.value = "";
        elements.showAll.checked = false;
        elements.search.value = "";
        elements.deleteConfiguration.classList.add("d-none");
        elements.removeApplication.classList.add("d-none");
        populateReferenceColumns();
        resetMappingFromReference();
        clearErrors();
    }

    function loadConfiguration(configuration) {
        currentConfigurationId = String(configuration.id);
        selectedColumns = new Set(configuration.columns || []);
        elements.name.value = configuration.name;
        elements.saved.value = configuration.id;
        elements.deleteConfiguration.classList.remove("d-none");
        elements.removeApplication.classList.toggle("d-none", !configuration.applied);

        const requiresAll = [...selectedColumns].some(
            (columnName) => !columnByName.get(columnName)?.recommended
        );
        elements.showAll.checked = requiresAll;
        const referenceName =
            [...selectedColumns].find((columnName) => columnByName.has(columnName)) ||
            availableColumns()[0]?.name ||
            "";
        populateReferenceColumns(referenceName);

        const detected = new Set(
            columnByName.get(referenceName)?.unique_values || []
        );
        elements.rows.replaceChildren();
        Object.entries(configuration.mapping)
            .sort((left, right) => Number(left[0]) - Number(right[0]))
            .forEach(([value, label]) => {
                createMappingRow(value, label, !detected.has(value));
            });
        renderColumnList();
        clearErrors();
    }

    function selectedIncompatibilities() {
        const result = [];
        selectedColumns.forEach((columnName) => {
            const column = columnByName.get(columnName);
            if (!column) return;
            const compatibility = compatibilityFor(column);
            if (!compatibility.compatible) {
                result.push({column, missing: compatibility.missing});
            }
        });
        return result;
    }

    function showCompatibilityWarning(incompatibilities) {
        const warning = elements.compatibilityWarning;
        warning.replaceChildren();
        warning.classList.remove("d-none");

        const title = document.createElement("strong");
        title.textContent = "Hay columnas seleccionadas con valores sin equivalencia.";
        const list = document.createElement("ul");
        list.className = "mb-3 mt-2";
        incompatibilities.forEach(({column, missing}) => {
            const item = document.createElement("li");
            item.textContent = `${column.name}: ${missing.join(", ")}`;
            list.append(item);
        });

        const actions = document.createElement("div");
        actions.className = "d-flex flex-wrap gap-2";
        const addMissing = document.createElement("button");
        addMissing.type = "button";
        addMissing.className = "btn btn-warning btn-sm";
        addMissing.textContent = "Agregar valores faltantes";
        addMissing.addEventListener("click", () => {
            const existing = configuredValues();
            const missingValues = new Set(
                incompatibilities.flatMap((item) => item.missing)
            );
            [...missingValues]
                .filter((value) => !existing.has(value))
                .sort((left, right) => Number(left) - Number(right))
                .forEach((value) => createMappingRow(value, "", true));
            sortMappingRows();
            warning.classList.add("d-none");
            renderColumnList();
        });

        const removeColumns = document.createElement("button");
        removeColumns.type = "button";
        removeColumns.className = "btn btn-outline-dark btn-sm";
        removeColumns.textContent = "Quitar columnas incompatibles";
        removeColumns.addEventListener("click", () => {
            incompatibilities.forEach(({column}) => selectedColumns.delete(column.name));
            warning.classList.add("d-none");
            renderColumnList();
        });

        const cancel = document.createElement("button");
        cancel.type = "button";
        cancel.className = "btn btn-outline-secondary btn-sm";
        cancel.textContent = "Cancelar";
        cancel.addEventListener("click", () => warning.classList.add("d-none"));
        actions.append(addMissing, removeColumns, cancel);
        warning.append(title, list, actions);
    }

    function validateClientForm() {
        let valid = true;
        const rows = mappingRows();
        const seen = new Set();

        if (!elements.name.value.trim()) {
            elements.name.classList.add("is-invalid");
            elements.nameError.textContent = "Escribe un nombre para la configuración.";
            valid = false;
        }
        if (!rows.length) {
            setFieldError(elements.mappingError, "Agrega al menos una equivalencia.");
            valid = false;
        }
        rows.forEach(({value, label, row}) => {
            const inputs = row.querySelectorAll("input");
            const canonical = canonicalClientNumber(value);
            const duplicate = canonical && seen.has(canonical);
            inputs[0].classList.toggle("is-invalid", !canonical || duplicate);
            inputs[1].classList.toggle("is-invalid", !label);
            if (!canonical || duplicate || !label) valid = false;
            if (canonical) seen.add(canonical);
        });
        if (!valid && rows.length) {
            setFieldError(
                elements.mappingError,
                "Completa los significados y evita valores cuantitativos repetidos."
            );
        }
        if (!selectedColumns.size) {
            setFieldError(elements.columnsError, "Selecciona al menos una columna.");
            valid = false;
        }
        return valid;
    }

    function renderServerErrors(errors) {
        if (errors.name) {
            elements.name.classList.add("is-invalid");
            elements.nameError.textContent = errors.name;
        }
        if (errors.equivalences || errors.equivalence_rows) {
            setFieldError(
                elements.mappingError,
                errors.equivalences || "Revisa las equivalencias indicadas."
            );
        }
        if (errors.columns) setFieldError(elements.columnsError, errors.columns);
        if (errors.incompatible_columns) {
            showAlert("Existen columnas con valores que aún no tienen equivalencia.");
        } else if (errors.form || errors.dataset || errors.configuration) {
            showAlert(errors.form || errors.dataset || errors.configuration);
        }
    }

    elements.reference.addEventListener("change", () => {
        renderReferenceSummary();
        resetMappingFromReference();
    });
    elements.showAll.addEventListener("change", () => {
        populateReferenceColumns();
        renderColumnList();
    });
    elements.search.addEventListener("input", renderColumnList);
    document.getElementById("addEquivalenceButton").addEventListener("click", () => {
        createMappingRow("", "", true);
        elements.rows.lastElementChild.querySelector("input").focus();
    });
    elements.toggleCompatible?.addEventListener("click", () => {
        const compatibleColumns = availableColumns().filter(
            (column) => compatibilityFor(column).compatible
        );
        const allCompatibleSelected =
            compatibleColumns.length > 0 &&
            compatibleColumns.every((column) => selectedColumns.has(column.name));
        if (allCompatibleSelected) {
            selectedColumns.clear();
        } else {
            compatibleColumns.forEach((column) => selectedColumns.add(column.name));
        }
        renderColumnList();
    });
    document.getElementById("newConfigurationButton").addEventListener(
        "click",
        initializeNewConfiguration
    );
    elements.saved.addEventListener("change", () => {
        const configuration = configurationById.get(elements.saved.value);
        if (configuration) loadConfiguration(configuration);
        else initializeNewConfiguration();
    });

    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        clearErrors();
        if (!validateClientForm()) return;
        const incompatibilities = selectedIncompatibilities();
        if (incompatibilities.length) {
            showCompatibilityWarning(incompatibilities);
            return;
        }

        const payload = {
            configuration_id: currentConfigurationId || null,
            name: elements.name.value.trim(),
            reference_column: elements.reference.value,
            equivalences: mappingRows().map(({value, label}) => ({value, label})),
            columns: [...selectedColumns],
        };
        elements.save.disabled = true;
        try {
            const response = await fetch(form.dataset.saveUrl, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "X-CSRFToken": csrfToken(),
                },
                body: JSON.stringify(payload),
            });
            const result = await response.json();
            if (!response.ok) {
                renderServerErrors(result.errors || {});
                return;
            }
            showAlert(result.message, "success");
            window.setTimeout(() => window.location.reload(), 800);
        } catch {
            showAlert("No fue posible guardar la configuración. Intenta nuevamente.");
        } finally {
            elements.save.disabled = false;
        }
    });

    elements.deleteConfiguration.addEventListener("click", async () => {
        if (!currentConfigurationId || !window.confirm("¿Eliminar esta configuración guardada?")) return;
        const response = await fetch(
            configurationUrl(form.dataset.deleteUrl, currentConfigurationId),
            {method: "POST", headers: {"X-CSRFToken": csrfToken()}}
        );
        if (response.ok) window.location.reload();
        else showAlert("No fue posible eliminar la configuración.");
    });

    elements.removeApplication.addEventListener("click", async () => {
        if (!currentConfigurationId || !window.confirm("¿Quitar esta configuración del dataset actual?")) return;
        const response = await fetch(
            configurationUrl(form.dataset.removeUrl, currentConfigurationId),
            {method: "POST", headers: {"X-CSRFToken": csrfToken()}}
        );
        if (response.ok) window.location.reload();
        else showAlert("No fue posible quitar la configuración.");
    });

    populateSavedConfigurations();
    initializeNewConfiguration();
})();
