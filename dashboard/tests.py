from django.test import TestCase
from django.urls import reverse

from datasets.models import Dataset
from kmeans.models import KMeansRun
from kmeans.services import train_kmeans


class DashboardViewTests(TestCase):
    def test_dashboard_is_the_home_page(self):
        response = self.client.get(reverse('dashboard:index'))

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'dashboard/index.html')
        self.assertContains(response, 'Datos cargados')

    def test_dashboard_does_not_accept_post_requests(self):
        response = self.client.post(reverse('dashboard:index'))

        self.assertEqual(response.status_code, 405)

    def test_saved_models_use_spacious_three_column_scrollable_card_grid(self):
        dataset = Dataset.objects.create(
            pk=1,
            source_name='modelos.csv',
            columns=['x', 'y'],
            records=[
                {'x': '0', 'y': '0'},
                {'x': '0.1', 'y': '0.1'},
                {'x': '5', 'y': '5'},
                {'x': '5.1', 'y': '5.1'},
            ],
        )
        train_kmeans(dataset, ['x', 'y'], 2)

        response = self.client.get(reverse('dashboard:index'))

        self.assertContains(
            response,
            'max-height: 42rem; overflow-y: auto; overscroll-behavior: contain;',
        )
        self.assertContains(
            response,
            'class="col-md-6 col-xl-4"',
            count=1,
        )
        self.assertContains(response, 'Todos los modelos')
        self.assertContains(response, 'K-Means')
        self.assertEqual(
            {item['algorithm'] for item in response.context['all_saved_models']},
            {'kmeans'},
        )

    def test_deleting_dataset_preserves_downloadable_model_catalogue(self):
        dataset = Dataset.objects.create(
            pk=1,
            source_name='modelos.csv',
            columns=['x', 'y'],
            records=[
                {'x': '0', 'y': '0'},
                {'x': '0.1', 'y': '0.1'},
                {'x': '5', 'y': '5'},
                {'x': '5.1', 'y': '5.1'},
            ],
        )
        kmeans_run = train_kmeans(dataset, ['x', 'y'], 2)
        self.client.post(reverse('datasets:delete'))

        self.assertFalse(Dataset.objects.exists())
        self.assertIsNone(KMeansRun.objects.get(pk=kmeans_run.pk).dataset)
        response = self.client.get(reverse('dashboard:index'))
        self.assertEqual(len(response.context['all_saved_models']), 1)
        self.assertEqual(response.context['saved_model_count'], 1)
        self.assertContains(
            self.client.get(reverse('kmeans:export', args=[kmeans_run.pk])),
            '"type": "kmeans"',
        )

    def test_current_filter_controls_compatibility_and_retraining(self):
        dataset = Dataset.objects.create(
            pk=1,
            source_name='categorias.csv',
            columns=['categoria', 'x'],
            records=[
                {'categoria': 'Agua', 'x': '0'},
                {'categoria': 'Agua', 'x': '0.1'},
                {'categoria': 'Agua', 'x': '5'},
                {'categoria': 'Agua', 'x': '5.1'},
                {'categoria': 'Tierra', 'x': '10'},
                {'categoria': 'Tierra', 'x': '10.1'},
            ],
        )
        run = train_kmeans(dataset, ['x'], 2, requested_category='agua')

        response = self.client.get(
            reverse('dashboard:index'), {'category': 'tierra'}
        )
        item = response.context['all_saved_models'][0]
        self.assertFalse(item['compatible'])
        self.assertContains(response, 'El filtro actual no coincide')

        self.client.post(
            reverse('kmeans:retrain', args=[run.pk]),
            {'category': 'tierra'},
        )
        self.assertEqual(KMeansRun.objects.count(), 1)
        self.assertIn('model_action_error', self.client.session)
