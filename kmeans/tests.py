from copy import deepcopy

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from django.urls import reverse

from datasets.models import Dataset
from datasets.model_validation import model_compatibility

from .exports import export_kmeans_run, import_kmeans_run
from .models import KMeansRun
from .services import (
    KMeansTrainingError,
    build_results_context,
    detect_categorical_columns,
    detect_numeric_columns,
    train_kmeans,
)


class KMeansRetrainingTests(TestCase):
    def setUp(self):
        self.dataset = Dataset.objects.create(
            pk=1, source_name='original.csv',
            columns=['categoria', 'x', 'y', 'etiqueta'],
            records=[
                {'categoria': 'Agua', 'x': '1', 'y': '1', 'etiqueta': 'A'},
                {'categoria': 'Agua', 'x': '1.2', 'y': '1.1', 'etiqueta': 'A'},
                {'categoria': 'Agua', 'x': '8', 'y': '8', 'etiqueta': 'B'},
                {'categoria': 'Agua', 'x': '8.2', 'y': '8.1', 'etiqueta': 'B'},
            ],
        )

    def test_persists_metadata_and_reusable_estimator_state(self):
        run = train_kmeans(
            self.dataset, ['x', 'y'], 2, 'agua', 'etiqueta',
            name='Zodiaco', topic='Signos zodiacales',
            description='Primera versión',
        )
        self.assertEqual(run.name, 'Zodiaco')
        self.assertEqual(run.topic, 'Signos zodiacales')
        self.assertEqual(len(run.preprocessing_state['mean']), 2)
        self.assertEqual(len(run.estimator_state['normalized_centroids']), 2)
        self.assertTrue(run.dataset_schema_fingerprint)
        self.assertTrue(run.training_config_fingerprint)

    def test_retrain_endpoint_creates_child_version_and_comparison(self):
        parent = train_kmeans(
            self.dataset, ['x', 'y'], 2, 'agua', 'etiqueta',
            name='Zodiaco',
        )
        parent.is_saved = True
        parent.save(update_fields=['is_saved'])
        self.dataset.records.extend([
            {'categoria': 'Agua', 'x': '1.1', 'y': '0.9', 'etiqueta': 'A'},
            {'categoria': 'Agua', 'x': '8.1', 'y': '8.3', 'etiqueta': 'B'},
        ])
        self.dataset.source_name = 'actualizado.csv'
        self.dataset.save()

        response = self.client.post(reverse('kmeans:retrain', args=[parent.pk]))

        self.assertEqual(response.status_code, 302)
        child = KMeansRun.objects.exclude(pk=parent.pk).get()
        self.assertEqual(child.parent_run, parent)
        self.assertEqual(child.version, 2)
        self.assertEqual(child.new_record_count, 2)
        self.assertEqual(child.estimator_state['parameters']['init'], 'previous_centroids')
        self.assertEqual(child.change_summary['current_sample_count'], 6)

    def test_draft_is_hidden_until_user_saves_it(self):
        run = train_kmeans(
            self.dataset, ['x', 'y'], 2, name='Borrador',
            save_immediately=False,
        )
        response = self.client.get(reverse('dashboard:index'))
        self.assertNotIn(run, response.context['all_kmeans_runs'])

        response = self.client.post(
            reverse('kmeans:save', args=[run.pk]),
            {'name': 'Modelo elegido', 'topic': 'Zodiaco', 'description': 'Útil'},
        )

        self.assertEqual(response.status_code, 302)
        run.refresh_from_db()
        self.assertTrue(run.is_saved)
        self.assertIsNotNone(run.saved_at)
        self.assertEqual(run.name, 'Modelo elegido')
        response = self.client.get(reverse('dashboard:index'))
        self.assertIn(run, response.context['all_kmeans_runs'])

    def test_filter_mismatch_is_explained(self):
        run = train_kmeans(self.dataset, ['x', 'y'], 2, 'agua')
        result = model_compatibility(
            self.dataset, run, requested_category='tierra'
        )
        self.assertFalse(result['compatible'])
        self.assertIn('filtro actual', result['reasons'][0])

    def test_schema_type_change_is_incompatible(self):
        run = train_kmeans(self.dataset, ['x', 'y'], 2)
        for record in self.dataset.records:
            record['x'] = 'texto'
        self.dataset.save(update_fields=['records'])

        result = model_compatibility(self.dataset, run)

        self.assertFalse(result['compatible'])
        self.assertTrue(
            any('cambió de tipo' in reason for reason in result['reasons'])
        )

    def test_compatible_import_accepts_an_expanded_dataset(self):
        run = train_kmeans(
            self.dataset, ['x', 'y'], 2, 'agua', 'etiqueta',
            requested_category_column='categoria',
        )
        payload = export_kmeans_run(run)
        self.dataset.records.extend([
            {'categoria': 'Agua', 'x': '1.1', 'y': '1', 'etiqueta': 'A'},
            {'categoria': 'Agua', 'x': '8.1', 'y': '8', 'etiqueta': 'B'},
        ])
        self.dataset.save(update_fields=['records'])

        imported = import_kmeans_run(
            self.dataset, payload, allow_compatible=True
        )

        self.assertEqual(imported.dataset, self.dataset)
        self.assertEqual(imported.dataset_fingerprint, payload['dataset_fingerprint'])
        compatibility = model_compatibility(self.dataset, imported)
        self.assertFalse(compatibility['exact'])
        self.assertTrue(compatibility['compatible'])


class KMeansServiceTests(TestCase):
    def test_detects_only_useful_numeric_columns(self):
        dataset = self.create_dataset(
            ['ID', 'Edad', 'Puntaje', 'Nombre', 'Constante'],
            [
                {
                    'ID': str(index),
                    'Edad': str(20 + index),
                    'Puntaje': str(index % 3),
                    'Nombre': f'Persona {index}',
                    'Constante': '1',
                }
                for index in range(1, 7)
            ],
        )

        columns = detect_numeric_columns(dataset, dataset.records)

        self.assertEqual(
            [column['name'] for column in columns],
            ['Edad', 'Puntaje'],
        )

    def test_detects_repeated_categorical_columns_generically(self):
        dataset = self.create_dataset(
            ['Nombre', 'Categoria', 'Escala'],
            [
                {'Nombre': 'Ana', 'Categoria': 'Agua', 'Escala': '1'},
                {'Nombre': 'Luis', 'Categoria': 'Aire', 'Escala': '1'},
                {'Nombre': 'Eva', 'Categoria': 'Agua', 'Escala': '2'},
                {'Nombre': 'Paz', 'Categoria': 'Aire', 'Escala': '2'},
            ],
        )

        columns = detect_categorical_columns(dataset, dataset.records)

        self.assertEqual(
            [column['name'] for column in columns],
            ['Categoria', 'Escala'],
        )

    def test_training_groups_similar_rows_and_persists_metrics(self):
        dataset = self.create_dataset(
            ['x', 'y'],
            [
                {'x': '1', 'y': '1'},
                {'x': '1.2', 'y': '0.8'},
                {'x': '0.8', 'y': '1.1'},
                {'x': '9', 'y': '9'},
                {'x': '9.2', 'y': '8.8'},
                {'x': '8.8', 'y': '9.1'},
            ],
        )

        run = train_kmeans(dataset, ['x', 'y'], 2)

        first_group = {
            assignment['cluster'] for assignment in run.assignments[:3]
        }
        second_group = {
            assignment['cluster'] for assignment in run.assignments[3:]
        }
        self.assertEqual(len(first_group), 1)
        self.assertEqual(len(second_group), 1)
        self.assertNotEqual(first_group, second_group)
        self.assertEqual(run.sample_count, 6)
        self.assertGreater(run.inertia, 0)
        self.assertGreater(run.silhouette, 0.8)
        self.assertEqual(sum(run.cluster_sizes.values()), 6)

    def test_training_is_reproducible(self):
        dataset = self.create_dataset(
            ['x', 'y'],
            [
                {'x': str(value), 'y': str(value)}
                for value in [1, 1.1, 1.2, 8, 8.1, 8.2]
            ],
        )

        first = train_kmeans(dataset, ['x', 'y'], 2)
        second = train_kmeans(dataset, ['x', 'y'], 2)

        self.assertEqual(
            [item['cluster'] for item in first.assignments],
            [item['cluster'] for item in second.assignments],
        )
        self.assertEqual(first.centroids, second.centroids)

    def test_missing_numeric_values_are_imputed_with_the_median(self):
        dataset = self.create_dataset(
            ['x', 'y'],
            [
                {'x': '1', 'y': '1'},
                {'x': '', 'y': '1.2'},
                {'x': '2', 'y': '1.1'},
                {'x': '8', 'y': '9'},
                {'x': '9', 'y': '8.8'},
            ],
        )

        run = train_kmeans(dataset, ['x', 'y'], 2)

        self.assertEqual(run.imputed_values['x']['count'], 1)
        self.assertEqual(run.imputed_values['x']['median'], 5.0)

    def test_filter_limits_training_and_preserves_original_row_numbers(self):
        dataset = self.create_dataset(
            ['categoria', 'x'],
            [
                {'categoria': 'Agua', 'x': '1'},
                {'categoria': 'Tierra', 'x': '100'},
                {'categoria': 'Agua', 'x': '2'},
                {'categoria': 'Tierra', 'x': '101'},
                {'categoria': 'Agua', 'x': '9'},
                {'categoria': 'Agua', 'x': '10'},
            ],
        )

        run = train_kmeans(dataset, ['x'], 2, 'agua')

        self.assertEqual(run.sample_count, 4)
        self.assertEqual(
            [assignment['row_number'] for assignment in run.assignments],
            [1, 3, 5, 6],
        )
        self.assertEqual(run.category_filter, 'agua')

    def test_optional_comparison_calculates_cluster_predominance(self):
        dataset = self.create_dataset(
            ['x', 'Grupo real'],
            [
                {'x': '1', 'Grupo real': 'Agua'},
                {'x': '1.1', 'Grupo real': 'agua'},
                {'x': '1.2', 'Grupo real': 'Tierra'},
                {'x': '9', 'Grupo real': 'Fuego'},
                {'x': '9.1', 'Grupo real': 'Fuego'},
                {'x': '9.2', 'Grupo real': 'Fuego'},
            ],
        )

        run = train_kmeans(
            dataset,
            ['x'],
            2,
            comparison_column='Grupo real',
        )

        self.assertEqual(run.comparison_column, 'Grupo real')
        self.assertEqual(
            run.comparison_values,
            ['Agua', 'Fuego', 'Tierra'],
        )
        self.assertEqual(run.overall_match_percentage, 83.33)
        self.assertEqual(
            run.cluster_comparison[0]['predominant_category'],
            'Agua',
        )
        self.assertEqual(
            run.cluster_comparison[0]['match_percentage'],
            66.67,
        )
        self.assertEqual(
            run.cluster_comparison[1]['predominant_category'],
            'Fuego',
        )
        self.assertEqual(
            run.cluster_comparison[1]['match_percentage'],
            100.0,
        )

    def test_training_without_comparison_keeps_only_internal_metrics(self):
        dataset = self.create_dataset(
            ['x'],
            [{'x': value} for value in ['1', '1.2', '8', '8.2']],
        )

        run = train_kmeans(dataset, ['x'], 2)

        self.assertEqual(run.comparison_column, '')
        self.assertEqual(run.cluster_comparison, [])
        self.assertIsNone(run.overall_match_percentage)

    def test_comparison_ignores_null_categories_in_match_percentage(self):
        dataset = self.create_dataset(
            ['x', 'Grupo'],
            [
                {'x': '1', 'Grupo': 'A'},
                {'x': '1.1', 'Grupo': ''},
                {'x': '9', 'Grupo': 'B'},
                {'x': '9.1', 'Grupo': 'B'},
            ],
        )

        run = train_kmeans(
            dataset,
            ['x'],
            2,
            comparison_column='Grupo',
        )

        self.assertEqual(run.comparison_valid_count, 3)
        self.assertEqual(run.overall_match_percentage, 100.0)
        self.assertEqual(
            sum(
                cluster['compared_count']
                for cluster in run.cluster_comparison
            ),
            3,
        )

    def test_comparison_column_cannot_also_train_the_model(self):
        dataset = self.create_dataset(
            ['x', 'Escala'],
            [
                {'x': '1', 'Escala': '1'},
                {'x': '2', 'Escala': '1'},
                {'x': '8', 'Escala': '2'},
                {'x': '9', 'Escala': '2'},
            ],
        )

        with self.assertRaisesMessage(
            KMeansTrainingError,
            'no puede utilizarse para entrenar',
        ):
            train_kmeans(
                dataset,
                ['x', 'Escala'],
                2,
                comparison_column='Escala',
            )

    def test_rejects_more_clusters_than_distinct_rows(self):
        dataset = self.create_dataset(
            ['x'],
            [{'x': value} for value in ['1', '1', '2', '2', '3']],
        )

        with self.assertRaisesMessage(
            KMeansTrainingError,
            'No existen suficientes combinaciones diferentes',
        ):
            train_kmeans(dataset, ['x'], 4)

    def test_result_assignments_are_paginated_by_twenty_five(self):
        dataset = self.create_dataset(
            ['x'],
            [{'x': str(value)} for value in range(60)],
        )
        train_kmeans(dataset, ['x'], 3)

        context = build_results_context(dataset, 2)

        self.assertEqual(context['kmeans_result_page'].number, 2)
        self.assertEqual(
            len(context['kmeans_result_page'].object_list),
            25,
        )

    def test_two_dimension_chart_uses_training_values_and_centroids(self):
        dataset = self.create_dataset(
            ['x', 'y'],
            [
                {'x': '1', 'y': '2'},
                {'x': '1.2', 'y': '2.2'},
                {'x': '8', 'y': '9'},
                {'x': '8.2', 'y': '9.2'},
            ],
        )
        run = train_kmeans(dataset, ['x', 'y'], 2)

        context = build_results_context(dataset)
        chart = context['kmeans_chart']
        plotted_points = [
            point
            for cluster in chart['clusters']
            for point in cluster['points']
        ]

        self.assertFalse(chart['projected'])
        self.assertEqual(chart['x_label'], 'x')
        self.assertEqual(chart['y_label'], 'y')
        self.assertEqual(
            {(point['x'], point['y']) for point in plotted_points},
            {(1.0, 2.0), (1.2, 2.2), (8.0, 9.0), (8.2, 9.2)},
        )
        self.assertEqual(
            [(item['x'], item['y']) for item in chart['centroids']],
            [
                (centroid['values'][0], centroid['values'][1])
                for centroid in run.centroids
            ],
        )

    def test_chart_uses_pca_only_for_more_than_two_training_columns(self):
        dataset = self.create_dataset(
            ['x', 'y', 'z'],
            [
                {'x': str(value), 'y': str(value * 2), 'z': str(value % 3)}
                for value in range(1, 9)
            ],
        )
        train_kmeans(dataset, ['x', 'y', 'z'], 2)

        chart = build_results_context(dataset)['kmeans_chart']

        self.assertTrue(chart['projected'])
        self.assertIn('Proyección PCA', chart['method'])
        self.assertIn('Componente principal 1', chart['x_label'])
        self.assertEqual(
            sum(len(cluster['points']) for cluster in chart['clusters']),
            8,
        )

    def test_cluster_profiles_match_persisted_assignments(self):
        dataset = self.create_dataset(
            ['x'],
            [{'x': value} for value in ['1', '1.1', '1.2', '9', '9.1']],
        )
        run = train_kmeans(dataset, ['x'], 2)

        profiles = build_results_context(dataset)['kmeans_cluster_profiles']

        self.assertEqual(
            [profile['size'] for profile in profiles],
            [
                run.cluster_sizes[str(cluster)]
                for cluster in range(1, run.cluster_count + 1)
            ],
        )
        self.assertAlmostEqual(
            sum(profile['percentage'] for profile in profiles),
            100.0,
            places=1,
        )
        self.assertTrue(
            all('promedio' in profile['characteristic'] for profile in profiles)
        )

    @staticmethod
    def create_dataset(columns, records):
        return Dataset.objects.create(
            pk=1,
            source_name='kmeans.csv',
            columns=columns,
            records=records,
        )


class KMeansViewTests(TestCase):
    def test_training_tab_requires_a_dataset(self):
        response = self.client.get(reverse('dashboard:index'))

        self.assertContains(response, 'No hay datos para entrenar')
        self.assertNotContains(response, 'Entrenar K-Means')

    def test_training_interface_lists_numeric_columns(self):
        self.create_dataset()

        response = self.client.get(reverse('dashboard:index'))

        self.assertContains(response, 'Cantidad de grupos')
        self.assertContains(response, 'Seleccionar todas')
        self.assertNotContains(response, 'id="clearKMeansColumns"')
        self.assertContains(response, 'Edad')
        self.assertContains(response, 'Puntaje')
        self.assertContains(response, 'Columna para comparar resultados')
        self.assertContains(response, 'Sin comparación')
        self.assertContains(response, 'Segmento')
        self.assertNotContains(response, 'Nombre</span>')

    def test_valid_post_trains_and_redirects_to_results(self):
        self.create_dataset()

        response = self.client.post(
            reverse('kmeans:train'),
            {
                'algorithm': 'kmeans',
                'cluster_count': 2,
                'columns': ['Edad', 'Puntaje'],
            },
        )

        self.assertEqual(
            response.url,
            f"{reverse('dashboard:index')}?results_view=kmeans#results-pane",
        )
        self.assertEqual(KMeansRun.objects.count(), 1)

        results = self.client.get(reverse('dashboard:index'))
        self.assertContains(results, 'Resultados de K-Means')
        self.assertContains(results, 'Coeficiente de silueta')
        self.assertContains(results, 'Inercia')
        self.assertContains(results, 'Calidad del agrupamiento')
        self.assertContains(results, 'Separación visual de los clusters')
        self.assertContains(results, 'Qué encontró el algoritmo')
        self.assertContains(results, 'kmeansClusterChart')
        self.assertNotContains(results, 'Composición de los clusters')
        self.assertNotContains(results, 'Asignación de registros')

    def test_training_with_comparison_renders_external_summary(self):
        self.create_dataset()

        self.client.post(
            reverse('kmeans:train'),
            {
                'algorithm': 'kmeans',
                'cluster_count': 2,
                'columns': ['Edad', 'Puntaje'],
                'comparison_column': 'Segmento',
            },
        )

        run = KMeansRun.objects.get()
        self.assertNotIn('Segmento', run.selected_columns)
        self.assertEqual(run.comparison_column, 'Segmento')

        response = self.client.get(reverse('dashboard:index'))
        self.assertContains(response, 'Comparación con Segmento')
        self.assertContains(response, 'Categoría predominante')
        self.assertContains(response, 'Coincidencia general')

    def test_post_without_columns_returns_to_training_with_error(self):
        self.create_dataset()

        response = self.client.post(
            reverse('kmeans:train'),
            {'algorithm': 'kmeans', 'cluster_count': 2},
            follow=True,
        )

        self.assertContains(response, 'Este campo es obligatorio.')
        self.assertFalse(KMeansRun.objects.exists())

    def test_replacing_dataset_preserves_previous_training_model(self):
        dataset = self.create_dataset()
        train_kmeans(dataset, ['Edad', 'Puntaje'], 2)
        new_file = SimpleUploadedFile(
            'nuevo.csv',
            b'x,y\n1,1\n2,2\n',
            content_type='text/csv',
        )

        self.client.post(reverse('datasets:upload'), {'file': new_file})

        self.assertTrue(KMeansRun.objects.exists())
        dashboard = self.client.get(reverse('dashboard:index'))
        self.assertIsNone(dashboard.context['kmeans_run'])
        self.assertContains(dashboard, 'Otro dataset')

    def test_incompatible_saved_model_cannot_be_activated_directly(self):
        dataset = self.create_dataset()
        run = train_kmeans(dataset, ['Edad', 'Puntaje'], 2)
        new_file = SimpleUploadedFile(
            'nuevo.csv',
            b'x,y\n1,1\n2,2\n',
            content_type='text/csv',
        )
        self.client.post(reverse('datasets:upload'), {'file': new_file})

        response = self.client.post(
            reverse('kmeans:activate', args=[run.pk]),
            follow=True,
        )

        self.assertContains(response, 'pertenece a otro dataset')
        self.assertEqual(response.request['PATH_INFO'], reverse('dashboard:index'))
        self.assertFalse(response.context['all_saved_models'][0]['compatible'])

    def test_results_show_reset_training_button(self):
        dataset = self.create_dataset()
        train_kmeans(dataset, ['Edad', 'Puntaje'], 2)

        response = self.client.get(reverse('dashboard:index'))

        self.assertContains(response, 'Reiniciar entrenamiento')
        self.assertContains(response, reverse('kmeans:reset'))

    def test_reset_removes_training_and_preserves_dataset(self):
        dataset = self.create_dataset()
        train_kmeans(
            dataset, ['Edad', 'Puntaje'], 2, save_immediately=False
        )

        response = self.client.post(reverse('kmeans:reset'))

        self.assertEqual(
            response.url,
            f"{reverse('dashboard:index')}#training-pane",
        )
        self.assertFalse(KMeansRun.objects.exists())
        self.assertTrue(Dataset.objects.filter(pk=dataset.pk).exists())

    @staticmethod
    def create_dataset():
        return Dataset.objects.create(
            pk=1,
            source_name='personas.csv',
            columns=['Nombre', 'Edad', 'Puntaje', 'Segmento'],
            records=[
                {'Nombre': 'A', 'Edad': '18', 'Puntaje': '1', 'Segmento': 'Joven'},
                {'Nombre': 'B', 'Edad': '19', 'Puntaje': '1.2', 'Segmento': 'Joven'},
                {'Nombre': 'C', 'Edad': '20', 'Puntaje': '1.1', 'Segmento': 'Adulto'},
                {'Nombre': 'D', 'Edad': '40', 'Puntaje': '9', 'Segmento': 'Adulto'},
                {'Nombre': 'E', 'Edad': '41', 'Puntaje': '9.2', 'Segmento': 'Adulto'},
                {'Nombre': 'F', 'Edad': '42', 'Puntaje': '8.9', 'Segmento': 'Adulto'},
            ],
        )


class KMeansImportValidationTests(TestCase):
    def setUp(self):
        self.dataset = Dataset.objects.create(
            pk=1,
            source_name='datos.csv',
            columns=['x', 'y'],
            records=[
                {'x': '0', 'y': '0'},
                {'x': '0.1', 'y': '0.1'},
                {'x': '5', 'y': '5'},
                {'x': '5.1', 'y': '5.1'},
            ],
        )
        self.run = train_kmeans(
            self.dataset,
            selected_columns=['x', 'y'],
            cluster_count=2,
        )
        self.payload = export_kmeans_run(self.run)

    def test_valid_export_import_round_trip(self):
        self.run.delete()

        imported = import_kmeans_run(self.dataset, self.payload)

        self.assertEqual(imported.assignments, self.payload['assignments'])
        self.assertEqual(build_results_context(self.dataset)['kmeans_run'], imported)

    def test_import_rejects_different_dataset(self):
        self.dataset.records[0]['x'] = '999'
        self.dataset.save(update_fields=['records'])

        with self.assertRaisesMessage(ValueError, 'conjunto de datos diferente'):
            import_kmeans_run(self.dataset, self.payload)

    def test_import_rejects_invalid_centroid_dimensions(self):
        payload = deepcopy(self.payload)
        payload['centroids'][0]['values'] = [1]

        with self.assertRaisesMessage(ValueError, 'dimensiones'):
            import_kmeans_run(self.dataset, payload)
