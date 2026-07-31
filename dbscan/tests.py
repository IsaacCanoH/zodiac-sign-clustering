import json
from copy import deepcopy

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from django.urls import reverse

from datasets.models import Dataset
from datasets.services import replace_dataset
from kmeans.services import train_kmeans

from .exports import export_dbscan_run, import_dbscan_run
from .models import DBSCANRun
from .services import (
    DBSCANTrainingError,
    build_dbscan_results_context,
    train_dbscan,
)


class DBSCANServiceTests(TestCase):
    def setUp(self):
        self.dataset = Dataset.objects.create(
            pk=1,
            source_name='grupos.csv',
            columns=['x', 'y', 'categoria'],
            records=[
                {'x': '0', 'y': '0', 'categoria': 'A'},
                {'x': '0.05', 'y': '0.04', 'categoria': 'A'},
                {'x': '0.1', 'y': '0', 'categoria': 'A'},
                {'x': '10', 'y': '10', 'categoria': 'B'},
                {'x': '10.05', 'y': '10.04', 'categoria': 'B'},
                {'x': '10.1', 'y': '10', 'categoria': 'B'},
                {'x': '30', 'y': '30', 'categoria': 'C'},
            ],
        )

    def train(self):
        return train_dbscan(
            dataset=self.dataset,
            selected_columns=['x', 'y'],
            epsilon=0.1,
            min_samples=2,
            comparison_column='categoria',
        )

    def test_training_finds_clusters_noise_and_real_metrics(self):
        run = self.train()

        self.assertEqual(run.sample_count, 7)
        self.assertEqual(run.cluster_count, 2)
        self.assertEqual(run.noise_count, 1)
        self.assertEqual(run.cluster_sizes, {'1': 3, '2': 3, '-1': 1})
        self.assertEqual(run.comparison_valid_count, 6)
        self.assertEqual(run.overall_match_percentage, 100)
        self.assertIsNotNone(run.silhouette)
        self.assertGreater(run.silhouette, 0.9)
        self.assertEqual(run.silhouette_sample_count, 6)

    def test_retraining_is_versioned_full_refit_with_saved_state(self):
        parent = self.train()
        parent.is_saved = True
        parent.save(update_fields=['is_saved'])
        self.dataset.records.extend([
            {'x': '0.08', 'y': '0.02', 'categoria': 'A'},
            {'x': '10.08', 'y': '10.02', 'categoria': 'B'},
        ])
        self.dataset.save(update_fields=['records'])

        response = self.client.post(reverse('dbscan:retrain', args=[parent.pk]))

        self.assertEqual(response.status_code, 302)
        child = DBSCANRun.objects.exclude(pk=parent.pk).get()
        self.assertEqual(child.parent_run, parent)
        self.assertEqual(child.version, 2)
        self.assertEqual(child.new_record_count, 2)
        self.assertEqual(child.estimator_state['strategy'], 'full_refit')
        self.assertIn('components', child.estimator_state)
        self.assertIn('changed_cluster_count', child.change_summary)

    def test_all_noise_is_reported_as_a_correctable_error(self):
        with self.assertRaisesMessage(
            DBSCANTrainingError,
            'no encontró ningún cluster',
        ):
            train_dbscan(
                dataset=self.dataset,
                selected_columns=['x', 'y'],
                epsilon=0.000001,
                min_samples=2,
            )

        self.assertFalse(DBSCANRun.objects.exists())

    def test_missing_values_are_imputed_with_training_median(self):
        self.dataset.records[1]['x'] = ''
        self.dataset.save(update_fields=['records'])

        run = self.train()

        self.assertEqual(run.imputed_values['x']['count'], 1)
        self.assertEqual(run.imputed_values['x']['median'], 10.025)

    def test_results_context_uses_original_rows_and_builds_chart(self):
        run = self.train()

        context = build_dbscan_results_context(self.dataset)

        self.assertEqual(context['dbscan_run'], run)
        self.assertEqual(len(context['dbscan_result_rows']), 7)
        self.assertEqual(context['dbscan_chart']['total_count'], 7)
        self.assertEqual(
            {group['cluster'] for group in context['dbscan_chart']['groups']},
            {-1, 1, 2},
        )
        self.assertEqual(context['dbscan_result_rows'][0]['values'], ['0', '0'])

    def test_replacing_dataset_preserves_models_but_hides_old_results(self):
        original_records = deepcopy(self.dataset.records)
        self.train()
        train_kmeans(
            dataset=self.dataset,
            selected_columns=['x', 'y'],
            cluster_count=2,
        )

        replace_dataset(
            {
                'file': SimpleUploadedFile('nuevo.csv', b'x,y\n1,2\n'),
                'columns': ['x', 'y'],
                'records': [{'x': '1', 'y': '2'}],
            }
        )

        self.assertTrue(DBSCANRun.objects.exists())
        self.assertTrue(self.dataset.kmeans_runs.exists())
        response = self.client.get(reverse('dashboard:index'))
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.context['dbscan_run'])
        self.assertIsNone(response.context['kmeans_run'])
        self.assertContains(response, 'Otro dataset', count=2)

        replace_dataset(
            {
                'file': SimpleUploadedFile('grupos.csv', b'contenido'),
                'columns': ['x', 'y', 'categoria'],
                'records': original_records,
            }
        )
        restored = self.client.get(reverse('dashboard:index'))
        self.assertIsNotNone(restored.context['dbscan_run'])
        self.assertIsNotNone(restored.context['kmeans_run'])
        self.assertTrue(
            all(
                item['compatible']
                for item in restored.context['all_saved_models']
            )
        )


class DBSCANImportTests(TestCase):
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
        self.run = train_dbscan(
            dataset=self.dataset,
            selected_columns=['x', 'y'],
            epsilon=0.1,
            min_samples=2,
        )
        self.payload = export_dbscan_run(self.run)

    def test_valid_export_import_round_trip(self):
        self.run.delete()

        imported = import_dbscan_run(self.dataset, self.payload)
        context = build_dbscan_results_context(self.dataset)

        self.assertEqual(imported.assignments, self.payload['assignments'])
        self.assertEqual(context['dbscan_run'], imported)

    def test_import_rejects_another_dataset_even_with_same_columns(self):
        payload = deepcopy(self.payload)
        self.dataset.records[0]['x'] = '999'
        self.dataset.save(update_fields=['records'])

        with self.assertRaisesMessage(ValueError, 'conjunto de datos diferente'):
            import_dbscan_run(self.dataset, payload)

    def test_import_rejects_inconsistent_assignments_without_saving(self):
        payload = deepcopy(self.payload)
        payload['assignments'][0]['row_number'] = 999
        initial_count = DBSCANRun.objects.count()

        with self.assertRaisesMessage(ValueError, 'fila que no existe'):
            import_dbscan_run(self.dataset, payload)

        self.assertEqual(DBSCANRun.objects.count(), initial_count)

    def test_import_view_handles_invalid_utf8_without_server_error(self):
        uploaded = SimpleUploadedFile(
            'modelo.json',
            b'\xff\xfe\x00',
            content_type='application/json',
        )

        response = self.client.post(
            reverse('dbscan:import'),
            {'model_file': uploaded},
        )

        self.assertRedirects(
            response,
            f"{reverse('dashboard:index')}#models-pane",
            fetch_redirect_response=False,
        )
        self.assertIn('model_import_error', self.client.session)

    def test_import_view_opens_dbscan_results(self):
        self.run.delete()
        uploaded = SimpleUploadedFile(
            'modelo.json',
            json.dumps(self.payload).encode(),
            content_type='application/json',
        )

        response = self.client.post(
            reverse('dbscan:import'),
            {'model_file': uploaded},
        )

        self.assertEqual(
            response.url,
            f"{reverse('dashboard:index')}?results_view=dbscan#results-pane",
        )


class DBSCANViewTests(TestCase):
    def setUp(self):
        self.dataset = Dataset.objects.create(
            pk=1,
            source_name='datos.csv',
            columns=['x'],
            records=[
                {'x': '0'},
                {'x': '0.1'},
                {'x': '5'},
                {'x': '5.1'},
            ],
        )

    def test_training_redirects_to_dbscan_result(self):
        response = self.client.post(
            reverse('dbscan:train'),
            {
                'epsilon': '0.1',
                'min_samples': '2',
                'columns': ['x'],
                'comparison_column': '',
            },
        )

        self.assertEqual(
            response.url,
            f"{reverse('dashboard:index')}?results_view=dbscan#results-pane",
        )
        self.assertEqual(DBSCANRun.objects.count(), 1)
        results = self.client.get(
            reverse('dashboard:index'),
            {'results_view': 'dbscan'},
        )
        self.assertEqual(results.status_code, 200)
        self.assertContains(results, 'Separación encontrada')
        self.assertContains(results, 'dbscanClusterChart')
        self.assertContains(results, 'dbscan-chart-data')
        self.assertContains(results, 'Configuración:')
        self.assertEqual(
            len(results.context['dbscan_run'].selected_columns),
            1,
        )
        self.assertNotContains(results, 'Columnas entrenadas')

    def test_training_uses_one_button_to_select_and_clear_columns(self):
        response = self.client.get(reverse('dashboard:index'))

        self.assertContains(response, 'id="toggleDbscanColumns"')
        self.assertContains(response, 'Seleccionar todas')
        self.assertNotContains(response, 'id="selectAllDbscanColumns"')
        self.assertNotContains(response, 'id="clearDbscanColumns"')

    def test_activation_preserves_creation_date_and_changes_active_run(self):
        first = train_dbscan(
            self.dataset,
            ['x'],
            epsilon=0.1,
            min_samples=2,
        )
        second = train_dbscan(
            self.dataset,
            ['x'],
            epsilon=0.2,
            min_samples=2,
        )
        original_created_at = first.created_at

        response = self.client.post(reverse('dbscan:activate', args=[first.pk]))
        first.refresh_from_db()

        self.assertEqual(first.created_at, original_created_at)
        self.assertGreater(first.activated_at, second.activated_at)
        self.assertEqual(self.dataset.dbscan_runs.first(), first)
        self.assertEqual(
            response.url,
            f"{reverse('dashboard:index')}?results_view=dbscan#results-pane",
        )
