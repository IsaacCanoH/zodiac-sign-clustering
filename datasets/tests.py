from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from django.urls import reverse

from .models import Dataset


class DatasetViewTests(TestCase):
    def test_csv_upload_persists_columns_and_records(self):
        csv_file = SimpleUploadedFile(
            'personas.csv',
            b'nombre,edad\nAna,28\nLuis,34\n',
            content_type='text/csv',
        )

        response = self.client.post(reverse('datasets:upload'), {'file': csv_file})

        self.assertRedirects(response, reverse('dashboard:index'))
        dataset = Dataset.objects.get(pk=1)
        self.assertEqual(dataset.columns, ['nombre', 'edad'])
        self.assertEqual(dataset.records[0], {'nombre': 'Ana', 'edad': '28'})

    def test_new_upload_replaces_previous_dataset(self):
        Dataset.objects.create(
            pk=1,
            source_name='anterior.csv',
            columns=['anterior'],
            records=[{'anterior': 'dato'}],
        )
        new_file = SimpleUploadedFile(
            'nuevo.csv', b'categoria,valor\nA,10\n', content_type='text/csv'
        )

        self.client.post(reverse('datasets:upload'), {'file': new_file})

        self.assertEqual(Dataset.objects.count(), 1)
        dataset = Dataset.objects.get(pk=1)
        self.assertEqual(dataset.source_name, 'nuevo.csv')
        self.assertEqual(dataset.columns, ['categoria', 'valor'])

    def test_invalid_file_does_not_replace_current_dataset(self):
        original = Dataset.objects.create(
            pk=1,
            source_name='actual.csv',
            columns=['valor'],
            records=[{'valor': '1'}],
        )
        invalid_file = SimpleUploadedFile(
            'datos.txt', b'valor\n2\n', content_type='text/plain'
        )

        response = self.client.post(
            reverse('datasets:upload'), {'file': invalid_file}
        )

        self.assertRedirects(
            response, f"{reverse('dashboard:index')}?upload=invalid"
        )
        original.refresh_from_db()
        self.assertEqual(original.source_name, 'actual.csv')

    def test_single_column_csv_is_accepted(self):
        csv_file = SimpleUploadedFile(
            'valores.csv', b'valor\nuno\ndos\n', content_type='text/csv'
        )

        response = self.client.post(reverse('datasets:upload'), {'file': csv_file})

        self.assertRedirects(response, reverse('dashboard:index'))
        self.assertEqual(Dataset.objects.get().row_count, 2)

    def test_dataset_table_paginates_twenty_five_rows(self):
        self.create_numbered_dataset(60)

        response = self.client.get(reverse('dashboard:index'), {'page': 2})

        self.assertEqual(response.context['page_obj'].number, 2)
        self.assertEqual(len(response.context['table_rows']), 25)
        self.assertEqual(response.context['table_rows'][0]['number'], 26)
        self.assertEqual(response.context['table_rows'][-1]['number'], 50)

    def test_out_of_range_page_returns_last_page(self):
        self.create_numbered_dataset(30)

        response = self.client.get(reverse('dashboard:index'), {'page': 99})

        self.assertEqual(response.context['page_obj'].number, 2)
        self.assertEqual(len(response.context['table_rows']), 5)

    def test_dataset_can_be_removed_manually(self):
        self.create_numbered_dataset(1)

        response = self.client.post(reverse('datasets:delete'))

        self.assertRedirects(response, reverse('dashboard:index'))
        self.assertFalse(Dataset.objects.exists())

    def test_pagination_links_return_to_dataset_table(self):
        self.create_numbered_dataset(30)

        response = self.client.get(reverse('dashboard:index'))

        self.assertContains(response, '?page=2#dataset-table')

    @staticmethod
    def create_numbered_dataset(row_count):
        return Dataset.objects.create(
            pk=1,
            source_name='datos.csv',
            columns=['valor'],
            records=[{'valor': str(number)} for number in range(row_count)],
        )
