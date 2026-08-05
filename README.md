# Aplicativo de análisis de datos y clasificación no supervisada

Aplicación web construida con Django para cargar conjuntos de datos en CSV o XLSX, explorarlos, transformar escalas numéricas a etiquetas cualitativas, obtener estadística descriptiva y entrenar modelos de agrupamiento **K-Means**.

Este README documenta la arquitectura, los módulos funcionales y la ubicación de las implementaciones principales, incluyendo el algoritmo de agrupamiento, la selección del número de grupos y el análisis estadístico.

## Mapa de funcionalidades principales

| Funcionalidad | Archivo principal | Implementación principal |
| --- | --- | --- |
| Punto de entrada y pantallas integradas | `dashboard/views.py` | `DashboardView.get_context_data` |
| Rutas generales | `config/urls.py` | `urlpatterns` |
| Lectura y validación de CSV/XLSX | `datasets/forms.py` | `DatasetUploadForm.clean`, `_read_csv`, `_read_excel` |
| Guardar, filtrar y paginar datos | `datasets/services.py` | `replace_dataset`, `filter_dataset_by_category`, `build_dataset_context` |
| Datos sintéticos | `datasets/services.py` | `generate_synthetic_records` |
| Equivalencias número → categoría | `datasets/equivalences.py` | `analyze_numeric_columns`, `validate_equivalence_payload`, `transform_records` |
| Estadística descriptiva | `descriptive_statistics/services.py` | `analyze_quantitative_column`, `analyze_qualitative_column` |
| Interpretación escrita de estadísticas | `descriptive_statistics/interpretations.py` | `build_statistical_interpretation` |
| **Recomendación de número de clusters** | `kmeans/services.py` | **`analyze_kmeans_candidates`** y `_elbow_candidate` |
| **Algoritmo K-Means definitivo** | `kmeans/services.py` | **`train_kmeans`** |
| Preparación: columnas, nulos y escalado | `kmeans/services.py` | `detect_numeric_columns`, `_build_matrix`, `StandardScaler` |
| Métricas y calidad del modelo | `kmeans/services.py` | `silhouette_score`, `davies_bouldin_score`, `calinski_harabasz_score` dentro de `analyze_kmeans_candidates` y `train_kmeans` |
| Predicción con un modelo ya guardado | `kmeans/services.py` | `predict_kmeans` |
| Resultados, perfiles y gráfico PCA | `kmeans/services.py` | `build_results_context`, `_cluster_profiles`, `_chart_context` |
| Exportar/importar modelos | `kmeans/exports.py` | `export_kmeans_run`, `import_kmeans_run` |
| Compatibilidad y reentrenamiento | `datasets/model_validation.py` | `model_compatibility`, `build_change_summary` |
| Datos ya clasificados por cluster | `classified_data/services.py` | `build_classified_records` |
| Persistencia en la base de datos | `datasets/models.py`, `kmeans/models.py` | `Dataset`, `EquivalenceConfiguration`, `KMeansRun` |

## Qué hace el sistema

1. El usuario carga un archivo `.csv` o `.xlsx`.
2. El sistema valida encabezados, filas y tamaño; guarda un único conjunto activo en SQLite.
3. Se puede filtrar el conjunto por una columna categórica, descargarlo o generar registros sintéticos para pruebas.
4. Se pueden configurar equivalencias como `1 → Bajo`, `2 → Medio`, `3 → Alto` sin modificar el valor original almacenado.
5. Se selecciona una variable para calcular estadística descriptiva, tabla de frecuencias, gráficas, correlación y reporte PDF.
6. Para K-Means se seleccionan las variables numéricas y se ejecuta primero un análisis de candidatos de `k` (número de clusters). El sistema propone un valor con silueta y método del codo.
7. El entrenamiento normaliza los datos, imputa nulos con la mediana, ejecuta K-Means, calcula métricas, genera perfiles y conserva el modelo para exportarlo, reactivarlo o reentrenarlo.

## Arquitectura y recorrido de una petición

```text
Navegador / plantilla / JavaScript
          │
          ▼
config/urls.py ──► views.py de cada módulo
                          │
                          ▼
                    services.py     ← reglas de negocio y algoritmos
                          │
             ┌────────────┼─────────────┐
             ▼            ▼             ▼
        models.py     exports.py    components.py
        (SQLite)      (PDF/XLSX/    (contexto para
                       JSON)         las plantillas)
```

La vista principal es `DashboardView` en `dashboard/views.py`. Esta no contiene el algoritmo: reúne los contextos construidos por los módulos de datos, estadísticas, K-Means y datos clasificados. La lógica importante está deliberadamente separada en los archivos `services.py`, lo cual facilita probarla y explicarla.

## Estructura del proyecto

```text
Aplicativo Datos/
├── config/                         # Configuración global de Django y rutas raíz
│   ├── settings.py
│   └── urls.py
├── dashboard/                      # Página principal que integra todos los paneles
│   ├── views.py
│   └── templates/dashboard/index.html
├── datasets/                       # Carga, almacenamiento, filtros y equivalencias
│   ├── forms.py
│   ├── models.py
│   ├── services.py
│   ├── equivalences.py
│   ├── model_validation.py
│   ├── exports.py
│   └── static/datasets/equivalences.js
├── descriptive_statistics/         # Estadística, interpretación, gráficas y PDF
│   ├── services.py
│   ├── interpretations.py
│   ├── exports.py
│   └── static/descriptive_statistics/charts.js
├── kmeans/                         # Entrenamiento, evaluación, modelos y resultados
│   ├── services.py                 # Núcleo del algoritmo
│   ├── views.py
│   ├── forms.py
│   ├── models.py
│   ├── exports.py
│   ├── pdf_reports.py
│   └── static/kmeans/workspace.js
├── classified_data/                # Tabla/descarga de registros con cluster asignado
│   └── services.py
├── requirements.txt
├── manage.py
└── db.sqlite3                      # Base de datos local de desarrollo
```

## Instalación y ejecución

Requiere Python y las dependencias declaradas en `requirements.txt` (Django, NumPy, pandas, scikit-learn, openpyxl y ReportLab).

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

Abrir `http://127.0.0.1:8000/` en el navegador. El panel de administración de Django está en `/admin/`.

## Rutas importantes

| URL | Módulo | Propósito |
| --- | --- | --- |
| `/` | `dashboard` | Dashboard y todos los paneles de trabajo. |
| `/datasets/cargar/` | `datasets` | Recibe el CSV/XLSX. |
| `/datasets/descargar/` | `datasets` | Descarga el conjunto filtrado en XLSX. |
| `/datasets/generar-sinteticos/` | `datasets` | Añade registros sintéticos de prueba. |
| `/datasets/equivalencias/guardar/` | `datasets` | Guarda una configuración de equivalencias. |
| `/estadistica/descargar-pdf/` | `descriptive_statistics` | Genera el reporte estadístico PDF. |
| `/kmeans/analizar-clusters/` | `kmeans` | Analiza valores candidatos de `k`, sin entrenar un modelo definitivo. |
| `/kmeans/entrenar/` | `kmeans` | Ejecuta el entrenamiento definitivo. |
| `/kmeans/<id>/exportar/` | `kmeans` | Exporta un modelo en JSON. |
| `/kmeans/importar/` | `kmeans` | Importa un modelo JSON validado. |
| `/kmeans/<id>/reentrenar/` | `kmeans` | Crea una nueva versión del modelo. |
| `/datos-clasificados/descargar/` | `classified_data` | Descarga registros con su cluster. |

Las rutas se declaran primero en `config/urls.py` y cada módulo registra sus endpoints en su propio `urls.py`.

## Módulo 1: carga y administración del dataset

### Lectura y validación del archivo

El procesamiento de archivos se implementa en `datasets/forms.py` mediante `DatasetUploadForm`.

- `clean_file`: acepta únicamente `.csv` y `.xlsx`, con máximo de 10 MB.
- `_read_csv`: detecta delimitador entre coma, punto y coma, tabulador o barra vertical.
- `_read_excel`: lee la hoja activa mediante `openpyxl`.
- `_normalize_dataset`: exige encabezados no vacíos y no duplicados; convierte cada fila en un diccionario `{columna: valor}`.

La vista `DatasetUploadView.post` de `datasets/views.py` recibe el formulario. Al ser válido llama a `replace_dataset` de `datasets/services.py`, que ejecuta `update_or_create(pk=1)`: la aplicación mantiene un solo dataset activo.

```python
# datasets/services.py — replace_dataset
with transaction.atomic():
    dataset = Dataset.objects.update_or_create(
        pk=1,
        defaults={
            'source_name': cleaned_data['file'].name,
            'columns': cleaned_data['columns'],
            'records': cleaned_data['records'],
        },
    )[0]
    dataset.equivalence_applications.all().delete()
```

### Persistencia del conjunto de datos

`datasets/models.py` define tres entidades:

| Modelo | Responsabilidad |
| --- | --- |
| `Dataset` | Nombre del archivo, lista JSON de columnas y lista JSON de registros. |
| `EquivalenceConfiguration` | Plantilla reutilizable de valores numéricos y sus etiquetas. |
| `DatasetEquivalenceApplication` | Relación entre una configuración y las columnas del dataset donde se aplica. |

La base de desarrollo es `db.sqlite3`. Los modelos K-Means se guardan por separado, de modo que el catálogo puede conservar modelos aunque se reemplace el archivo de datos.

### Filtro por categoría y paginación

En `datasets/services.py`, `filter_dataset_by_category` detecta columnas de baja cardinalidad (máximo 50 valores distintos), prioriza nombres como `categoría`, `clase` o `label` y devuelve solo las filas que coinciden con el filtro elegido. `build_dataset_context` transforma la representación, pagina a 25 filas y prepara la tabla.

## Módulo 2: equivalencias cuantitativas a cualitativas

Este módulo permite conservar el valor numérico original y mostrar otra representación con significado humano. Por ejemplo, una escala `1, 2, 3` puede visualizarse como `Bajo, Medio, Alto`.

El núcleo está en `datasets/equivalences.py`:

- `canonical_number`: normaliza valores numéricos para compararlos sin ambigüedad (`1.0` se vuelve `1`).
- `analyze_numeric_columns`: detecta columnas numéricas y recomienda escalas enteras con pocos valores; evita identificadores, fechas y datos continuos.
- `validate_equivalence_payload`: confirma que toda columna seleccionada tenga equivalencia para todos sus valores observados.
- `transform_records`: construye una vista `original`, `quantitative` o `qualitative`, **sin alterar los registros originales**.

```python
# datasets/equivalences.py — criterio de recomendación
recommended = (
    integer_ratio >= Decimal('0.90')
    and reduced_cardinality
    and not possible_identifier
    and not possible_date
)
```

La interacción del modal está en `datasets/static/datasets/equivalences.js`; el servidor vuelve a validar todo en `EquivalenceSaveView` de `datasets/views.py`, por lo que no se depende solamente del navegador.

## Módulo 3: generación de datos sintéticos

Ubicación: `datasets/services.py`, función `generate_synthetic_records`.

Su objetivo es ampliar el conjunto para pruebas, no reemplazar datos reales. Selecciona un registro de referencia y:

- Para valores numéricos calcula mínimo, máximo y desviación estándar; agrega ruido gaussiano proporcional y limita el resultado al rango original.
- Para categorías conserva la distribución observada y ocasionalmente selecciona una categoría ponderada por frecuencia.
- Añade `Marca temporal` y `Tipo de registro = Sintético (Pruebas)` para identificarlos.

```python
noise = random.gauss(0, stats['std_dev'] * noise_level)
new_val = val + noise
new_val = max(stats['min'], min(stats['max'], new_val))
```

## Módulo 4: estadística descriptiva

El archivo principal es `descriptive_statistics/services.py`. La función de orquestación es `build_statistics_context`: aplica primero el filtro de categoría, detecta columnas válidas, elige el tipo de análisis y prepara los datos de las gráficas.

### Columnas y análisis

| Función | Qué hace |
| --- | --- |
| `detect_statistical_columns` | Separa variables cuantitativas y cualitativas y excluye nombres que parecen identificadores. |
| `analyze_quantitative_column` | Calcula media, mediana, moda, mínimo, máximo, rango, varianza, desviación estándar y frecuencias. |
| `_grouped_frequency_rows` | Para datos continuos con muchos valores usa la regla de Sturges y forma intervalos. |
| `analyze_qualitative_column` | Calcula moda, número de categorías, frecuencias absolutas, relativas, porcentajes y acumulados. |
| `_scatter_analysis` | Obtiene pares de valores y correlación para la gráfica de dispersión. |

```python
# descriptive_statistics/services.py — métricas cuantitativas
'metrics': {
    'mean': _round_number(statistics.mean(values)),
    'mode': mode_display,
    'median': _round_number(statistics.median(values)),
    'minimum': _round_number(min(values)),
    'maximum': _round_number(max(values)),
    'range': _round_number(max(values) - min(values)),
    'variance': _round_number(statistics.pvariance(values)),
    'standard_deviation': _round_number(statistics.pstdev(values)),
},
```

### Interpretaciones y gráficas

- `descriptive_statistics/interpretations.py`: `build_statistical_interpretation` transforma las métricas en una explicación legible; no inventa significados del dominio.
- `descriptive_statistics/static/descriptive_statistics/charts.js`: dibuja barras, líneas y dispersión con Chart.js.
- `descriptive_statistics/exports.py`: `build_statistics_pdf` crea el reporte PDF con ReportLab.

## Módulo 5: algoritmo K-Means

### Ubicación y propósito

La implementación principal del algoritmo se encuentra en `kmeans/services.py`, en la función `train_kmeans`. Antes del entrenamiento, `analyze_kmeans_candidates` evalúa distintos valores de `k` y produce una recomendación a partir de la silueta y el método del codo.

### Preparación de los datos

También en `kmeans/services.py`:

1. `detect_numeric_columns` conserva solo columnas numéricas no constantes y excluye identificadores como ID, folio, teléfono o matrícula.
2. `_build_matrix` convierte las filas seleccionadas en una matriz NumPy e **imputa los valores nulos con la mediana** de su columna. Conserva las medianas utilizadas.
3. `StandardScaler` estandariza las variables antes de agruparlas, de forma que una variable con escala grande no domine a las demás.

```python
# kmeans/services.py — imputación con mediana en _build_matrix
median = float(statistics.median(available))
for row in matrix:
    if row[column_index] is None:
        row[column_index] = median

# dentro de train_kmeans
scaler = StandardScaler()
scaled_matrix = scaler.fit_transform(matrix)
```

### Recomendación automática: ¿cuántos clusters usar?

**Archivo:** `kmeans/services.py`
**Función:** `analyze_kmeans_candidates`

Esta función no guarda un modelo final. Evalúa desde `k=1` hasta un máximo seguro (10, menor que las filas disponibles y menor que las combinaciones distintas). Para cada candidato ejecuta K-Means y registra:

- `inertia`: suma de distancias cuadradas internas; sirve para el método del codo.
- `silhouette`: separación interna de los grupos; mientras más alto, mejor separación.
- `davies_bouldin`: mide similitud entre clusters; menor suele ser mejor.
- `calinski_harabasz`: relación entre dispersión entre/dentro de clusters; mayor suele ser mejor.
- tamaño del cluster más pequeño y advertencias si representa menos del 2 %.

La recomendación principal es el `k` con mejor silueta, eligiendo el menor `k` que queda a menos de 0.01 del máximo. `_elbow_candidate` normaliza los pares `(k, inertia)` y elige el punto más alejado de la recta entre los extremos: ese es el codo.

```python
# kmeans/services.py — idea central de la recomendación
best_silhouette = max(item['silhouette'] for item in eligible)
recommended_silhouette = min(
    item['k'] for item in eligible
    if item['silhouette'] >= best_silhouette - 0.01
)
recommended_elbow = _elbow_candidate(results)
```

La petición web la maneja `KMeansAnalysisView.post` en `kmeans/views.py`. El resultado temporal se conserva en sesión y se presenta antes del entrenamiento definitivo.

### Entrenamiento definitivo

**Archivo:** `kmeans/services.py`
**Función:** `train_kmeans`

Esta función central recibe el dataset, las columnas, el número de clusters, el filtro opcional y una columna categórica opcional para comparación. Sus etapas son:

1. Valida que las variables sean numéricas y que `2 ≤ k < número de filas`.
2. Construye e imputa la matriz; verifica que existan suficientes filas distintas.
3. Estandariza con `StandardScaler`.
4. Ejecuta `sklearn.cluster.KMeans` con `k-means++`, 10 inicializaciones, máximo 300 iteraciones y semilla fija `42` para reproducibilidad.
5. Devuelve los centroides a su escala original, ordena los labels de forma estable y registra la asignación de cada fila.
6. Calcula métricas, advertencias, estabilidad, comparación externa y persiste un `KMeansRun`.

```python
# kmeans/services.py — instancia y ejecución real del algoritmo
estimator = KMeans(
    n_clusters=cluster_count,
    init='k-means++',
    n_init=10,
    max_iter=300,
    random_state=42,
    algorithm='lloyd',
)
labels = estimator.fit_predict(scaled_matrix)
original_centers = scaler.inverse_transform(estimator.cluster_centers_)
```

La vista que lo invoca es `KMeansTrainingView.post` en `kmeans/views.py`. En el flujo actual el resultado se crea primero como provisional (`save_immediately=False`) y `KMeansSaveView.post` lo promueve al catálogo al guardar nombre, tema y descripción.

### Métricas, advertencias y estabilidad

El mismo `train_kmeans` calcula:

| Elemento | Uso |
| --- | --- |
| Silhouette | Evalúa qué tan separados están los clusters. |
| Inercia | Compactación interna; se muestra junto con la recomendación. |
| Davies-Bouldin y Calinski-Harabasz | Diagnóstico interno adicional. |
| Pureza, ARI, NMI, homogeneidad, completitud y V-measure | Se calculan si el usuario proporciona una columna categórica de comparación. |
| `quality_warnings` | Advierte clusters menores a 2 %, imputación alta, silueta baja o límite de iteraciones. |
| `stability_metrics` | Compara la solución con cinco inicializaciones independientes mediante Adjusted Rand Index (ARI). |

La comparación por categoría se construye en `_comparison_summary`; no se usa como característica de entrenamiento, pues una columna de comparación no puede ser una de las columnas seleccionadas.

### Resultados, perfiles y visualización

| Ubicación | Responsabilidad |
| --- | --- |
| `kmeans/services.py` → `build_results_context` | Construye tabla paginada, resúmenes y contexto completo de resultados. |
| `kmeans/services.py` → `_cluster_profiles` | Describe qué variable diferencia más a cada cluster respecto al promedio general. |
| `kmeans/services.py` → `_chart_context` | Prepara puntos y centroides. Con 1–2 variables usa los valores originales; con más de 2 proyecta a dos componentes mediante PCA. |
| `kmeans/static/kmeans/workspace.js` | Dibuja el diagrama de dispersión, centroides y tooltips con Chart.js. |
| `kmeans/pdf_reports.py` → `build_kmeans_results_pdf` | Genera el reporte PDF de resultados. |

### Predicción con un modelo guardado

`predict_kmeans` en `kmeans/services.py` aplica el estado almacenado del modelo sin volver a entrenar: utiliza las medianas de imputación, medias y escalas originales, calcula la distancia a cada centroide y asigna el cluster más cercano.

```python
# kmeans/services.py — asignación al centroide más próximo
normalized = (np.asarray(matrix, dtype=float) - means) / safe_scales
distances = np.linalg.norm(
    normalized[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2
)
labels = np.argmin(distances, axis=1)
```

## Módulo 6: persistencia, catálogo, importación y reentrenamiento

`kmeans/models.py` define `KMeansRun`. Este modelo conserva, entre otros datos:

- variables usadas, cantidad de clusters, asignaciones, centroides y tamaños;
- métricas internas/externas, advertencias y estabilidad;
- valores usados en la imputación y estado del escalador, para reproducir predicciones;
- huellas del dataset y de su esquema, configuración de entrenamiento y versiones de librerías;
- nombre, tema, descripción, modelo padre y versión para el reentrenamiento.

La seguridad de reutilización está en `datasets/model_validation.py`:

- `dataset_fingerprint` calcula SHA-256 sobre columnas y registros para identificar un dataset exacto.
- `build_schema_profile` y `model_compatibility` verifican columnas, tipos y filtro antes de activar o reentrenar un modelo.
- `build_change_summary` compara versión anterior y nueva: filas compartidas, filas nuevas, cambios de cluster y desplazamiento de centroides.

Los serializadores validados están en `kmeans/exports.py`: `export_kmeans_run` produce el JSON e `import_kmeans_run` valida su formato antes de restaurarlo. Las vistas HTTP correspondientes están en `kmeans/views.py` (`KMeansExportView`, `KMeansImportView`, `KMeansActivateView` y `KMeansRetrainView`).

## Módulo 7: datos clasificados

Una vez entrenado un modelo activo, `classified_data/services.py` relaciona las asignaciones guardadas con las filas originales.

- `_active_run` obtiene el entrenamiento compatible con el dataset actual.
- `build_classified_records` agrega a cada registro su número de fila y etiqueta `Cluster N`, y permite filtrar por cluster.
- `build_classified_context` pagina los resultados.
- `classified_data/views.py` y `datasets/exports.py` generan el XLSX descargable.

## Interfaz: plantillas y JavaScript

Las plantillas HTML no contienen cálculos estadísticos ni el algoritmo; muestran los datos preparados en las capas anteriores.

| Archivo | Rol |
| --- | --- |
| `dashboard/templates/dashboard/index.html` | Layout general, pestañas y composición de los espacios de trabajo. |
| `datasets/templates/datasets/workspace.html` | Tabla del dataset, filtros y acciones de carga/descarga. |
| `descriptive_statistics/templates/descriptive_statistics/workspace.html` | Panel de selección y resultados estadísticos. |
| `kmeans/templates/kmeans/training_workspace.html` | Formulario de análisis y entrenamiento. |
| `kmeans/templates/kmeans/results_workspace.html` | Métricas, perfiles, tabla y visualización del modelo. |
| `classified_data/templates/classified_data/workspace.html` | Tabla de registros clasificados. |

Los gráficos se renderizan en el navegador usando Chart.js a partir de JSON preparado por los servicios del servidor.

## Pruebas automatizadas

Las pruebas se encuentran junto a cada módulo en `tests.py`:

- `datasets/tests.py`: carga, errores, paginación, filtros, equivalencias y datos sintéticos.
- `descriptive_statistics/tests.py`: métricas, frecuencias, interpretación, filtros, correlación y PDF.
- `kmeans/tests.py`: selección asistida de `k`, reproducibilidad, imputación, métricas, estabilidad, resultados, exportación/importación y reentrenamiento.
- `classified_data/tests.py`: visualización y descarga por cluster.

Para ejecutarlas:

```powershell
python manage.py test
```

## Límites y consideraciones metodológicas

- K-Means necesita variables numéricas y funciona mejor con grupos aproximadamente compactos; una silueta baja se reporta como advertencia, no como certeza de clasificación.
- Los clusters son agrupaciones descubiertas, no etiquetas reales. Las métricas externas solo tienen sentido si se proporciona una categoría de referencia.
- Las equivalencias cambian la representación visual, no el dato original ni las variables usadas para K-Means.
- Los datos sintéticos son adecuados para demostraciones y pruebas; no deben presentarse como observaciones reales.
