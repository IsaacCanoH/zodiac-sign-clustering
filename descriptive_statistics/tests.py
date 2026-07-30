from django.test import RequestFactory, TestCase
from django.urls import reverse

from datasets.models import (
    Dataset,
    DatasetEquivalenceApplication,
    EquivalenceConfiguration,
)

from .services import (
    analyze_qualitative_column,
    analyze_quantitative_column,
    build_statistics_context,
    detect_statistical_columns,
)


class DescriptiveStatisticsTests(TestCase):
    def setUp(self):
        self.request_factory = RequestFactory()

    def test_quantitative_analysis_calculates_requested_metrics(self):
        dataset = self.create_dataset(
            ['Escala'],
            [{'Escala': value} for value in ['1', '2', '2', '3', '4', '5']],
        )

        analysis = analyze_quantitative_column(
            dataset, dataset.records, 'Escala'
        )

        self.assertEqual(analysis['metrics']['mean'], 2.8333)
        self.assertEqual(analysis['metrics']['median'], 2.5)
        self.assertEqual(analysis['metrics']['mode'], '2.0')
        self.assertEqual(analysis['metrics']['minimum'], 1.0)
        self.assertEqual(analysis['metrics']['maximum'], 5.0)
        self.assertEqual(analysis['metrics']['range'], 4.0)
        self.assertEqual(analysis['metrics']['variance'], 1.8056)
        self.assertEqual(analysis['metrics']['standard_deviation'], 1.3437)

    def test_frequency_table_uses_saved_qualitative_equivalences(self):
        dataset = self.create_dataset(
            ['Escala'],
            [{'Escala': value} for value in ['1', '1', '2']],
        )
        configuration = EquivalenceConfiguration.objects.create(
            name='Acuerdo',
            mapping={'1': 'En desacuerdo', '2': 'De acuerdo'},
            possible_values=['1', '2'],
        )
        DatasetEquivalenceApplication.objects.create(
            dataset=dataset,
            configuration=configuration,
            columns=['Escala'],
        )

        analysis = analyze_quantitative_column(
            dataset, dataset.records, 'Escala'
        )

        self.assertEqual(
            analysis['frequency_rows'],
            [
                {
                    'value': '1',
                    'label': 'En desacuerdo',
                    'frequency': 2,
                    'relative_frequency': 0.6667,
                    'percentage': 66.67,
                    'cumulative_frequency': 2,
                },
                {
                    'value': '2',
                    'label': 'De acuerdo',
                    'frequency': 1,
                    'relative_frequency': 0.3333,
                    'percentage': 33.33,
                    'cumulative_frequency': 3,
                },
            ],
        )

    def test_qualitative_analysis_calculates_mode_and_frequencies(self):
        records = [
            {'Color': 'Azul'},
            {'Color': 'Rojo'},
            {'Color': 'Azul'},
            {'Color': ''},
        ]

        analysis = analyze_qualitative_column(records, 'Color')

        self.assertEqual(analysis['mode'], 'Azul')
        self.assertEqual(analysis['unique_count'], 2)
        self.assertEqual(analysis['valid_count'], 3)
        self.assertEqual(analysis['null_count'], 1)
        self.assertEqual(analysis['frequency_rows'][0]['frequency'], 2)
        self.assertEqual(analysis['frequency_rows'][0]['relative_frequency'], 0.6667)
        self.assertEqual(analysis['modes'], ['Azul'])
        self.assertEqual(analysis['mode_frequency'], 2)
        self.assertEqual(analysis['mode_percentage'], 66.67)

    def test_quantitative_interpretation_describes_center_and_dispersion(self):
        dataset = self.create_dataset(
            ['Puntaje'],
            [{'Puntaje': value} for value in ['1', '2', '2', '3', '4', '5']],
        )

        analysis = analyze_quantitative_column(
            dataset, dataset.records, 'Puntaje'
        )
        interpretation = ' '.join(analysis['interpretation'])

        self.assertIn('La media es 2.8333', interpretation)
        self.assertIn('La desviación estándar es 1.3437', interpretation)
        self.assertIn('El valor o conjunto de valores', interpretation)

    def test_qualitative_interpretation_identifies_a_dominant_category(self):
        analysis = analyze_qualitative_column(
            [
                {'Color': 'Azul'},
                {'Color': 'Azul'},
                {'Color': 'Azul'},
                {'Color': 'Rojo'},
            ],
            'Color',
        )
        interpretation = ' '.join(analysis['interpretation'])

        self.assertIn('“Azul”, con 3 registros (75.0%', interpretation)
        self.assertIn('concentración marcada', interpretation)

    def test_qualitative_interpretation_recognizes_multiple_modes(self):
        analysis = analyze_qualitative_column(
            [{'Color': 'Azul'}, {'Color': 'Rojo'}],
            'Color',
        )

        self.assertIn(
            'la distribución es multimodal',
            ' '.join(analysis['interpretation']),
        )

    def test_identifiers_are_excluded_but_age_is_available(self):
        dataset = self.create_dataset(
            ['ID', 'Matrícula', 'Teléfono', 'Edad'],
            [
                {
                    'ID': str(index),
                    'Matrícula': str(1000 + index),
                    'Teléfono': str(5550000 + index),
                    'Edad': str(18 + index % 4),
                }
                for index in range(12)
            ],
        )

        columns = detect_statistical_columns(dataset, dataset.records)

        self.assertEqual([column['name'] for column in columns], ['Edad'])
        self.assertEqual(columns[0]['kind'], 'quantitative')

    def test_continuous_data_uses_grouped_histogram(self):
        dataset = self.create_dataset(
            ['Ingreso'],
            [{'Ingreso': str(value * 1000)} for value in range(1, 31)],
        )

        analysis = analyze_quantitative_column(
            dataset, dataset.records, 'Ingreso'
        )

        self.assertTrue(analysis['chart']['grouped'])
        self.assertTrue(analysis['grouped_frequencies'])
        self.assertLessEqual(len(analysis['frequency_rows']), 15)
        self.assertLessEqual(len(analysis['chart']['labels']), 15)

    def test_discrete_scale_keeps_one_frequency_row_per_value(self):
        dataset = self.create_dataset(
            ['Escala'],
            [{'Escala': str(value)} for value in [1, 2, 3, 4, 5] * 4],
        )

        analysis = analyze_quantitative_column(
            dataset, dataset.records, 'Escala'
        )

        self.assertFalse(analysis['grouped_frequencies'])
        self.assertEqual(
            [row['value'] for row in analysis['frequency_rows']],
            ['1', '2', '3', '4', '5'],
        )

    def test_category_filter_is_applied_before_statistics(self):
        dataset = self.create_dataset(
            ['categoria', 'Puntaje'],
            [
                {'categoria': 'Agua', 'Puntaje': '1'},
                {'categoria': 'Agua', 'Puntaje': '3'},
                {'categoria': 'Tierra', 'Puntaje': '100'},
            ],
        )
        request = self.request_factory.get(
            '/', {'category': 'agua', 'stats_column': 'Puntaje'}
        )

        context = build_statistics_context(request, dataset)

        self.assertEqual(context['statistics_analysis']['valid_count'], 2)
        self.assertEqual(context['statistics_analysis']['metrics']['mean'], 2.0)

    def test_similar_scales_are_suggested_for_scatter_but_income_is_not(self):
        dataset = self.create_dataset(
            ['Escala A', 'Escala B', 'Ingreso'],
            [
                {
                    'Escala A': str(value),
                    'Escala B': str(max(value, 2)),
                    'Ingreso': str(value * 900000),
                }
                for value in range(1, 6)
            ],
        )
        request = self.request_factory.get(
            '/',
            {'stats_column': 'Escala A', 'compare_column': 'Escala B'},
        )

        context = build_statistics_context(request, dataset)
        candidates = {
            column['name']: column['suggested']
            for column in context['comparison_candidates']
        }

        self.assertTrue(candidates['Escala B'])
        self.assertFalse(candidates['Ingreso'])
        self.assertEqual(context['scatter_analysis']['pair_count'], 5)

    def test_statistics_interface_does_not_render_charts_without_dataset(self):
        response = self.client.get(reverse('dashboard:index'))

        self.assertContains(response, 'No hay datos para analizar')
        self.assertNotContains(response, 'primaryStatisticsChart')
        self.assertNotContains(response, 'Descargar PDF')

    def test_statistics_interface_renders_results_after_column_selection(self):
        self.create_dataset(
            ['Escala'],
            [{'Escala': value} for value in ['1', '2', '3', '4', '5']],
        )

        response = self.client.get(
            reverse('dashboard:index'), {'stats_column': 'Escala'}
        )

        self.assertContains(response, 'Medidas descriptivas')
        self.assertContains(response, 'primaryStatisticsChart')
        self.assertNotContains(response, 'Diagrama de caja')
        self.assertContains(response, 'Descargar PDF')

    def test_selected_statistics_can_be_downloaded_as_pdf(self):
        self.create_dataset(
            ['Escala'],
            [{'Escala': value} for value in ['1', '2', '2', '3', '4', '5']],
        )

        response = self.client.get(
            reverse('descriptive_statistics:download_pdf'),
            {'stats_column': 'Escala'},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'application/pdf')
        self.assertIn('attachment;', response['Content-Disposition'])
        self.assertTrue(response.content.startswith(b'%PDF'))

    def test_pdf_download_requires_a_valid_selected_column(self):
        self.create_dataset(['Escala'], [{'Escala': '1'}])

        response = self.client.get(
            reverse('descriptive_statistics:download_pdf')
        )

        self.assertEqual(response.status_code, 404)

    @staticmethod
    def create_dataset(columns, records):
        return Dataset.objects.create(
            pk=1,
            source_name='estadistica.csv',
            columns=columns,
            records=records,
        )
