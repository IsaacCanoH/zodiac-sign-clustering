from io import BytesIO
from html import escape

from reportlab.graphics.shapes import Circle, Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .services import (
    _chart_context,
    _cluster_profiles,
    _result_matrix,
    _silhouette_interpretation,
)


CLUSTER_COLORS = [
    colors.HexColor('#0057B8'),
    colors.HexColor('#E66100'),
    colors.HexColor('#009E73'),
    colors.HexColor('#8E44AD'),
    colors.HexColor('#D7191C'),
    colors.HexColor('#00A6D6'),
    colors.HexColor('#B28A00'),
    colors.HexColor('#CC79A7'),
]


def _styles():
    base = getSampleStyleSheet()
    return {
        'title': ParagraphStyle(
            'KMeansTitle', parent=base['Title'], fontName='Helvetica-Bold',
            fontSize=21, leading=25, textColor=colors.black, spaceAfter=3 * mm,
        ),
        'subtitle': ParagraphStyle(
            'KMeansSubtitle', parent=base['Normal'], fontSize=9.5,
            leading=13, textColor=colors.black,
        ),
        'section': ParagraphStyle(
            'KMeansSection', parent=base['Heading2'], fontName='Helvetica-Bold',
            fontSize=12.5, leading=16, textColor=colors.black,
            spaceBefore=5 * mm, spaceAfter=2.5 * mm,
        ),
        'body': ParagraphStyle(
            'KMeansBody', parent=base['BodyText'], fontSize=9,
            leading=12.5, textColor=colors.black,
        ),
        'center': ParagraphStyle(
            'KMeansCenter', parent=base['Normal'], fontSize=8.5,
            leading=11, textColor=colors.black, alignment=TA_CENTER,
        ),
        'header': ParagraphStyle(
            'KMeansHeader', parent=base['Normal'], fontName='Helvetica-Bold',
            fontSize=8, leading=10, textColor=colors.white,
            alignment=TA_CENTER,
        ),
    }


def _p(value, style):
    display = '-' if value in (None, '') else value
    return Paragraph(escape(str(display)), style)


def _footer(canvas, document):
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.black)
    canvas.setLineWidth(0.8)
    canvas.rect(7 * mm, 7 * mm, width - 14 * mm, height - 14 * mm)
    canvas.setLineWidth(0.35)
    canvas.line(18 * mm, 15 * mm, width - 18 * mm, 15 * mm)
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(colors.black)
    canvas.drawString(18 * mm, 10 * mm, 'Reporte de resultados K-Means')
    canvas.drawRightString(width - 18 * mm, 10 * mm, f'Página {document.page}')
    canvas.restoreState()


def _table(headings, rows, widths, styles, align_from=1):
    data = [[_p(value, styles['header']) for value in headings]]
    data.extend([[_p(value, styles['body']) for value in row] for row in rows])
    table = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.black),
        ('BOX', (0, 0), (-1, -1), 0.75, colors.black),
        ('INNERGRID', (0, 0), (-1, -1), 0.35, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (align_from, 1), (-1, -1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 2 * mm),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2 * mm),
    ]))
    return table


def _metric_table(run, quality, styles):
    values = [
        ('Clusters', run.cluster_count),
        ('Registros', run.sample_count),
        ('Silueta', f'{run.silhouette:.4f}' if run.silhouette is not None else '-'),
        ('Inercia', f'{run.inertia:.4f}'),
    ]
    cells = [
        Table(
            [[_p(label, styles['center'])],
             [_p(value, ParagraphStyle(
                 f'Metric{index}', parent=styles['center'],
                 fontName='Helvetica-Bold', fontSize=14, leading=17,
             ))]],
            colWidths=[42.5 * mm],
        )
        for index, (label, value) in enumerate(values)
    ]
    table = Table([cells], colWidths=[43.5 * mm] * 4)
    table.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 0.75, colors.black),
        ('INNERGRID', (0, 0), (-1, -1), 0.35, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 2 * mm),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2 * mm),
    ]))
    return [
        table,
        Spacer(1, 2.5 * mm),
        _p(
            f'Calidad del agrupamiento: {quality["label"]}. '
            f'{quality["description"]}',
            styles['body'],
        ),
    ]


def _cluster_chart(chart):
    width, height = 500, 285
    left, bottom, plot_width, plot_height = 58, 50, 405, 185
    drawing = Drawing(width, height)
    points = [
        point for cluster in chart['clusters'] for point in cluster['points']
    ]
    centroids = chart['centroids']
    all_x = [point['x'] for point in points] + [item['x'] for item in centroids]
    all_y = [point['y'] for point in points] + [item['y'] for item in centroids]
    min_x, max_x = min(all_x, default=0), max(all_x, default=1)
    min_y, max_y = min(all_y, default=0), max(all_y, default=1)
    if min_x == max_x:
        min_x, max_x = min_x - 1, max_x + 1
    if min_y == max_y:
        min_y, max_y = min_y - 1, max_y + 1
    x_pad = (max_x - min_x) * 0.08
    y_pad = (max_y - min_y) * 0.08
    min_x, max_x = min_x - x_pad, max_x + x_pad
    min_y, max_y = min_y - y_pad, max_y + y_pad

    def px(value):
        return left + (value - min_x) * plot_width / (max_x - min_x)

    def py(value):
        return bottom + (value - min_y) * plot_height / (max_y - min_y)

    drawing.add(Rect(left, bottom, plot_width, plot_height,
                     strokeColor=colors.black, fillColor=colors.white,
                     strokeWidth=0.8))
    for index in range(6):
        x = left + index * plot_width / 5
        y = bottom + index * plot_height / 5
        drawing.add(Line(x, bottom, x, bottom + plot_height,
                         strokeColor=colors.HexColor('#D9D9D9'), strokeWidth=0.35))
        drawing.add(Line(left, y, left + plot_width, y,
                         strokeColor=colors.HexColor('#D9D9D9'), strokeWidth=0.35))
        drawing.add(String(x, bottom - 13, f'{min_x + index * (max_x-min_x)/5:.2f}',
                           fontName='Helvetica', fontSize=6.5, textAnchor='middle'))
        drawing.add(String(left - 7, y - 2, f'{min_y + index * (max_y-min_y)/5:.2f}',
                           fontName='Helvetica', fontSize=6.5, textAnchor='end'))

    for cluster in chart['clusters']:
        color = CLUSTER_COLORS[(cluster['cluster'] - 1) % len(CLUSTER_COLORS)]
        for point in cluster['points']:
            drawing.add(Circle(px(point['x']), py(point['y']), 3,
                               fillColor=color, strokeColor=color))
    for centroid in centroids:
        color = CLUSTER_COLORS[(centroid['cluster'] - 1) % len(CLUSTER_COLORS)]
        drawing.add(Circle(px(centroid['x']), py(centroid['y']), 6,
                           fillColor=color, strokeColor=colors.black,
                           strokeWidth=1.5))

    drawing.add(String(left + plot_width / 2, 18, chart['x_label'][:65],
                       fontName='Helvetica-Bold', fontSize=8, textAnchor='middle'))
    if chart['y_label']:
        drawing.add(String(
            left, 242, f'Eje Y: {chart["y_label"][:60]}',
            fontName='Helvetica-Bold', fontSize=7.5,
        ))
    for index, cluster in enumerate(chart['clusters']):
        color = CLUSTER_COLORS[(cluster['cluster'] - 1) % len(CLUSTER_COLORS)]
        legend_x = left + (index % 5) * 80
        legend_y = 272 - (index // 5) * 12
        drawing.add(Circle(legend_x + 4, legend_y, 3.5,
                           fillColor=color, strokeColor=color))
        drawing.add(String(legend_x + 11, legend_y - 2.5,
                           f'Cluster {cluster["cluster"]}',
                           fontName='Helvetica', fontSize=7.5))
    return drawing


def build_kmeans_results_pdf(dataset, run):
    matrix = _result_matrix(dataset, run)
    chart = _chart_context(run, matrix)
    profiles = _cluster_profiles(run, matrix)
    quality = _silhouette_interpretation(run.silhouette)
    styles = _styles()
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=17 * mm, bottomMargin=20 * mm,
        title='Resultados de K-Means', author='Aplicativo de análisis de datos',
    )
    context_rows = [
        ['Dataset', run.dataset_source_name],
        ['Fecha de entrenamiento', run.created_at.strftime('%d/%m/%Y %H:%M')],
        [
            'Variables',
            f'{len(run.selected_columns)} variables seleccionadas '
            '(ver Anexo A: variables y orden).',
        ],
        ['Filtro aplicado', run.category_label or 'Todos los registros'],
        ['Método de visualización', chart['method']],
        ['Iteraciones', run.estimator_state.get('iterations', '-')],
        ['Semilla', run.estimator_state.get('parameters', {}).get('random_state', '-')],
    ]
    profile_rows = [
        [f'Cluster {row["cluster"]}', row['size'], f'{row["percentage"]:.2f}%',
         row['characteristic']]
        for row in profiles
    ]
    story = [
        Paragraph('Reporte de resultados K-Means', styles['title']),
        Paragraph(
            'Resumen del entrenamiento, separación visual y características de los clusters.',
            styles['subtitle'],
        ),
        Spacer(1, 5 * mm),
        Paragraph('1. Información general', styles['section']),
        _table(['Dato', 'Resultado'], context_rows, [47 * mm, 127 * mm], styles),
        Paragraph('2. Resumen del entrenamiento', styles['section']),
        *_metric_table(run, quality, styles),
        KeepTogether([
            Paragraph('3. Separación visual de los clusters', styles['section']),
            _p(
                f'{chart["method"]} Los puntos usan un color diferente por cluster; '
                'los centroides se muestran con borde negro.',
                styles['body'],
            ),
            Spacer(1, 2 * mm),
            _cluster_chart(chart),
        ]),
        Paragraph('4. Perfiles encontrados', styles['section']),
        _table(
            ['Cluster', 'Registros', 'Porcentaje', 'Característica principal'],
            profile_rows, [27 * mm, 24 * mm, 27 * mm, 96 * mm], styles,
        ),
    ]
    if run.comparison_column:
        comparison_rows = [
            [
                f'Cluster {row["cluster"]}', row['predominant_category'],
                row['compared_count'], f'{row["match_percentage"]:.2f}%',
            ]
            for row in run.cluster_comparison
        ]
        story.extend([
            Paragraph(
                f'5. Comparación con {escape(run.comparison_column)}',
                styles['section'],
            ),
            _p(
                'Pureza global: '
                + (f'{run.overall_match_percentage:.2f}%'
                   if run.overall_match_percentage is not None else 'No disponible'),
                styles['body'],
            ),
            Spacer(1, 2 * mm),
            _table(
                ['Cluster', 'Categoría predominante', 'Comparados', 'Coincidencia'],
                comparison_rows, [30 * mm, 72 * mm, 34 * mm, 38 * mm], styles,
            ),
            _p(
                'La pureza describe predominancia por cluster y no equivale a '
                'precisión de un clasificador supervisado. '
                f'ARI: {run.external_metrics.get("adjusted_rand", "-")}.',
                styles['body'],
            ),
        ])
    if run.results_by_k:
        diagnostic_rows = [
            [item['k'], f'{item["inertia"]:.2f}',
             '-' if item.get('silhouette') is None else f'{item["silhouette"]:.4f}']
            for item in run.results_by_k
        ]
        story.extend([
            Paragraph('6. Selección del número de clusters', styles['section']),
            _p(
                f'k seleccionado: {run.cluster_count}; mejor silueta: '
                f'{run.recommended_k_silhouette or "-"}; codo: '
                f'{run.recommended_k_elbow or "no concluyente"}.',
                styles['body'],
            ),
            Spacer(1, 2 * mm),
            _table(
                ['k', 'Inercia', 'Silueta'],
                diagnostic_rows, [35 * mm, 70 * mm, 69 * mm], styles,
                align_from=0,
            ),
        ])
    variable_rows = [
        [index, column]
        for index, column in enumerate(run.selected_columns, start=1)
    ]
    story.extend([
        PageBreak(),
        Paragraph('Anexo A. Variables y orden de entrenamiento', styles['section']),
        _p(
            'Esta lista conserva el orden exacto de las variables usado para entrenar el modelo.',
            styles['body'],
        ),
        Spacer(1, 2 * mm),
        _table(
            ['#', 'Variable'], variable_rows, [15 * mm, 159 * mm], styles,
            align_from=0,
        ),
    ])
    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buffer.getvalue()
