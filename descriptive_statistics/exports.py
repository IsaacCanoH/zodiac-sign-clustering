from io import BytesIO
from html import escape

from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.legends import Legend
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


PRIMARY = colors.black
SECONDARY = colors.black
INK = colors.black
MUTED = colors.black
SURFACE = colors.white
GRID = colors.black
WHITE = colors.white


def _styles():
    styles = getSampleStyleSheet()
    return {
        'title': ParagraphStyle(
            'ReportTitle',
            parent=styles['Title'],
            fontName='Helvetica-Bold',
            fontSize=22,
            leading=27,
            textColor=PRIMARY,
            spaceAfter=4 * mm,
        ),
        'subtitle': ParagraphStyle(
            'ReportSubtitle',
            parent=styles['Normal'],
            fontSize=10,
            leading=14,
            textColor=MUTED,
        ),
        'section': ParagraphStyle(
            'ReportSection',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=13,
            leading=17,
            textColor=PRIMARY,
            spaceBefore=5 * mm,
            spaceAfter=2.5 * mm,
        ),
        'body': ParagraphStyle(
            'ReportBody',
            parent=styles['BodyText'],
            fontSize=9.5,
            leading=13,
            textColor=INK,
        ),
        'metric_label': ParagraphStyle(
            'MetricLabel',
            parent=styles['Normal'],
            fontSize=9,
            leading=11,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
        'metric_value': ParagraphStyle(
            'MetricValue',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=14,
            leading=17,
            textColor=PRIMARY,
            alignment=TA_CENTER,
        ),
        'metric_value_long': ParagraphStyle(
            'MetricValueLong',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=10,
            leading=12,
            textColor=PRIMARY,
            alignment=TA_CENTER,
        ),
        'table_header': ParagraphStyle(
            'TableHeader',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=8,
            leading=10,
            textColor=WHITE,
            alignment=TA_CENTER,
        ),
        'table_cell': ParagraphStyle(
            'TableCell',
            parent=styles['Normal'],
            fontSize=8.5,
            leading=11,
            textColor=INK,
        ),
        'table_number': ParagraphStyle(
            'TableNumber',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=9,
            leading=11,
            textColor=INK,
            alignment=TA_CENTER,
        ),
        'interpretation': ParagraphStyle(
            'Interpretation',
            parent=styles['BodyText'],
            fontName='Helvetica',
            fontSize=10,
            leading=15,
            textColor=INK,
            spaceAfter=3 * mm,
        ),
    }


def _text(value):
    return str(value if value not in (None, '') else '-')


def _paragraph(value, style):
    return Paragraph(escape(_text(value)), style)


def _footer(canvas, document):
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.black)
    canvas.setLineWidth(1)
    canvas.rect(
        7 * mm,
        7 * mm,
        width - 14 * mm,
        height - 14 * mm,
        stroke=1,
        fill=0,
    )
    canvas.setLineWidth(0.4)
    canvas.line(18 * mm, 15 * mm, width - 18 * mm, 15 * mm)
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 10 * mm, 'Reporte de estadística descriptiva')
    canvas.drawRightString(
        width - 18 * mm,
        10 * mm,
        f'Página {document.page}',
    )
    canvas.restoreState()


def _metric_cards(analysis, styles):
    if analysis['kind'] == 'quantitative':
        metrics = [
            ('Media', analysis['metrics']['mean']),
            ('Mediana', analysis['metrics']['median']),
            ('Moda', analysis['metrics']['mode']),
            ('Rango', analysis['metrics']['range']),
            ('Mínimo', analysis['metrics']['minimum']),
            ('Máximo', analysis['metrics']['maximum']),
            ('Varianza poblacional', analysis['metrics']['variance']),
            ('Desviación estándar', analysis['metrics']['standard_deviation']),
        ]
        column_count = 4
    else:
        metrics = [
            ('Moda', analysis['mode']),
            ('Categorías distintas', analysis['unique_count']),
            ('Registros válidos', analysis['valid_count']),
        ]
        column_count = 3

    cells = []
    for label, value in metrics:
        value_style = (
            styles['metric_value_long']
            if len(_text(value)) > 18
            else styles['metric_value']
        )
        cells.append(
            [
                _paragraph(label, styles['metric_label']),
                _paragraph(value, value_style),
            ]
        )
    cell_width = (174 / column_count) * mm
    rows = []
    for start in range(0, len(cells), column_count):
        rows.append(
            [
                Table(
                    [[cell[0]], [cell[1]]],
                    colWidths=[cell_width - 4 * mm],
                    style=[
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('TOPPADDING', (0, 0), (-1, -1), 2 * mm),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 2 * mm),
                    ],
                )
                for cell in cells[start:start + column_count]
            ]
        )
    table = Table(rows, colWidths=[cell_width], hAlign='LEFT')
    table.setStyle(
        TableStyle(
            [
                ('BACKGROUND', (0, 0), (-1, -1), WHITE),
                ('BOX', (0, 0), (-1, -1), 0.8, GRID),
                ('INNERGRID', (0, 0), (-1, -1), 0.4, GRID),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 1.5 * mm),
                ('RIGHTPADDING', (0, 0), (-1, -1), 1.5 * mm),
                ('TOPPADDING', (0, 0), (-1, -1), 2 * mm),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 2 * mm),
            ]
        )
    )
    return table


def _bar_chart(chart):
    labels = [_text(label) for label in chart['labels']]
    needs_extra_label_space = len(labels) > 8 or any(
        len(label) > 8 for label in labels
    )
    drawing = Drawing(500, 235)
    drawing.add(
        String(
            20,
            215,
            'Distribución de frecuencias',
            fontName='Helvetica-Bold',
            fontSize=11,
            fillColor=INK,
        )
    )
    graph = VerticalBarChart()
    graph.x = 55
    graph.y = 85 if needs_extra_label_space else 48
    graph.height = 120 if needs_extra_label_space else 147
    graph.width = 410
    graph.data = [chart['frequencies']]
    graph.categoryAxis.categoryNames = labels
    graph.categoryAxis.labels.fontName = 'Helvetica'
    graph.categoryAxis.labels.fontSize = 7
    graph.categoryAxis.labels.angle = 55 if needs_extra_label_space else 0
    graph.categoryAxis.labels.boxAnchor = (
        'ne' if needs_extra_label_space else 'n'
    )
    graph.categoryAxis.labels.dy = -10
    graph.valueAxis.valueMin = 0
    graph.valueAxis.valueStep = max(1, max(chart['frequencies'], default=1) // 4)
    graph.valueAxis.labels.fontName = 'Helvetica'
    graph.valueAxis.labels.fontSize = 8
    graph.valueAxis.labels.dx = -9
    graph.bars[0].fillColor = colors.black
    graph.bars[0].strokeColor = colors.black
    drawing.add(graph)
    return drawing


def _secondary_chart(analysis):
    chart = analysis['chart']
    drawing = Drawing(500, 235)
    if analysis['kind'] == 'qualitative':
        drawing.add(
            String(
                20,
                215,
                'Participación por categoría',
                fontName='Helvetica-Bold',
                fontSize=11,
                fillColor=INK,
            )
        )
        pie = Pie()
        pie.x = 35
        pie.y = 20
        pie.width = 180
        pie.height = 180
        pie.data = chart['frequencies']
        pie.labels = None
        pie.slices.fontName = 'Helvetica'
        pie.slices.fontSize = 8
        palette = [
            colors.HexColor('#1F4E79'),
            colors.HexColor('#C55A11'),
            colors.HexColor('#548235'),
            colors.HexColor('#7030A0'),
            colors.HexColor('#BF9000'),
            colors.HexColor('#7F7F7F'),
        ]
        for index in range(len(pie.data)):
            pie.slices[index].fillColor = palette[index % len(palette)]
        drawing.add(pie)
        legend = Legend()
        legend.x = 245
        legend.y = 190
        legend.dx = 8
        legend.dy = 8
        legend.deltay = 12
        legend.fontName = 'Helvetica'
        legend.fontSize = 7.5
        legend.alignment = 'right'
        legend.columnMaximum = max(len(chart['labels']), 1)
        legend.colorNamePairs = [
            (pie.slices[index].fillColor, _text(label))
            for index, label in enumerate(chart['labels'])
        ]
        drawing.add(legend)
    else:
        drawing.add(
            String(
                20,
                215,
                'Polígono de frecuencias',
                fontName='Helvetica-Bold',
                fontSize=11,
                fillColor=INK,
            )
        )
        graph = HorizontalLineChart()
        graph.x = 55
        graph.y = 45
        graph.height = 150
        graph.width = 410
        graph.data = [chart['frequencies']]
        graph.categoryAxis.categoryNames = [
            _text(label)[:10] for label in chart['labels']
        ]
        graph.categoryAxis.labels.fontName = 'Helvetica'
        graph.categoryAxis.labels.fontSize = 7.5
        graph.categoryAxis.labels.angle = 0
        graph.categoryAxis.labels.dy = -12
        graph.valueAxis.valueMin = 0
        graph.valueAxis.labels.fontName = 'Helvetica'
        graph.valueAxis.labels.fontSize = 8
        graph.valueAxis.labels.dx = -9
        graph.lines[0].strokeColor = colors.black
        graph.lines[0].strokeWidth = 2
        drawing.add(graph)
    return drawing


def _frequency_table(analysis, styles):
    grouped = analysis.get('grouped_frequencies', False)
    first_heading = (
        'Intervalo'
        if grouped
        else ('Valor' if analysis['kind'] == 'quantitative' else 'Categoría')
    )
    headings = [first_heading]
    show_equivalence = analysis['kind'] == 'quantitative' and not grouped
    if show_equivalence:
        headings.append('Equivalencia')
    headings.extend(
        ['Frecuencia', 'Frecuencia relativa', 'Frecuencia acumulada', 'Porcentaje']
    )
    data = [
        [_paragraph(heading, styles['table_header']) for heading in headings]
    ]
    for row in analysis['frequency_rows']:
        values = [_text(row['value'])]
        if show_equivalence:
            values.append(_text(row.get('label')))
        values.extend(
            [
                _text(row['frequency']),
                _text(row['relative_frequency']),
                _text(row['cumulative_frequency']),
                f"{row['percentage']}%",
            ]
        )
        description_count = 2 if show_equivalence else 1
        data.append(
            [
                _paragraph(
                    value,
                    styles['table_cell']
                    if index < description_count
                    else styles['table_number'],
                )
                for index, value in enumerate(values)
            ]
        )

    first_width = 35 * mm if show_equivalence else 48 * mm
    widths = [first_width]
    if show_equivalence:
        widths.append(39 * mm)
    remaining_width = 174 * mm - sum(widths)
    widths.extend([remaining_width / 4] * 4)
    table = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
    table.setStyle(
        TableStyle(
            [
                ('BACKGROUND', (0, 0), (-1, 0), colors.black),
                ('BOX', (0, 0), (-1, -1), 0.8, GRID),
                ('INNERGRID', (0, 0), (-1, -1), 0.4, GRID),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('BACKGROUND', (0, 1), (-1, -1), WHITE),
                ('ALIGN', (2 if show_equivalence else 1, 1), (-1, -1), 'CENTER'),
                ('TOPPADDING', (0, 0), (-1, -1), 2.5 * mm),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5 * mm),
            ]
        )
    )
    return table


def build_statistics_pdf(dataset, analysis, category_label=''):
    buffer = BytesIO()
    styles = _styles()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=20 * mm,
        title=f'Estadística descriptiva - {analysis["column"]}',
        author='Aplicativo de análisis de datos',
    )
    variable_type = (
        'Cuantitativa' if analysis['kind'] == 'quantitative' else 'Cualitativa'
    )
    raw_context_rows = [
        ['Dataset', dataset.source_name],
        ['Columna analizada', analysis['column']],
        ['Tipo de variable', variable_type],
        ['Filtro aplicado', category_label or 'Todos los registros'],
        ['Datos válidos', analysis['valid_count']],
        ['Datos nulos', analysis['null_count']],
    ]
    context_rows = [
        [label, _paragraph(value, styles['body'])]
        for label, value in raw_context_rows
    ]
    context_table = Table(context_rows, colWidths=[40 * mm, 134 * mm])
    context_table.setStyle(
        TableStyle(
            [
                ('BACKGROUND', (0, 0), (0, -1), colors.black),
                ('TEXTCOLOR', (0, 0), (0, -1), WHITE),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9.5),
                ('BOX', (0, 0), (-1, -1), 0.8, GRID),
                ('INNERGRID', (0, 0), (-1, -1), 0.4, GRID),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 2 * mm),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 2 * mm),
            ]
        )
    )

    interpretation_content = []
    for paragraph in analysis['interpretation']:
        interpretation_content.extend(
            [_paragraph(paragraph, styles['interpretation']), Spacer(1, 1 * mm)]
        )
    interpretation_box = Table(
        [[interpretation_content]],
        colWidths=[174 * mm],
        hAlign='LEFT',
    )
    interpretation_box.setStyle(
        TableStyle(
            [
                ('BOX', (0, 0), (-1, -1), 0.8, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 5 * mm),
                ('RIGHTPADDING', (0, 0), (-1, -1), 5 * mm),
                ('TOPPADDING', (0, 0), (-1, -1), 5 * mm),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3 * mm),
            ]
        )
    )

    story = [
        Paragraph('Reporte de estadística descriptiva', styles['title']),
        Paragraph(
            'Resumen profesional de los resultados generados para la variable seleccionada.',
            styles['subtitle'],
        ),
        Spacer(1, 5 * mm),
        Paragraph('1. Información general', styles['section']),
        context_table,
        Paragraph('2. Medidas descriptivas', styles['section']),
        _metric_cards(analysis, styles),
        Paragraph('3. Tabla de frecuencias', styles['section']),
        Paragraph(
            (
                'Los valores se presentan por intervalos para facilitar su lectura.'
                if analysis.get('grouped_frequencies')
                else 'Los resultados se presentan individualmente para cada valor o categoría.'
            ),
            styles['body'],
        ),
        Spacer(1, 3 * mm),
        _frequency_table(analysis, styles),
        KeepTogether(
            [
                Paragraph('4. Visualizaciones', styles['section']),
                _bar_chart(analysis['chart']),
            ]
        ),
        Spacer(1, 8 * mm),
        _secondary_chart(analysis),
        KeepTogether(
            [
                Paragraph('5. Interpretación de resultados', styles['section']),
                interpretation_box,
            ]
        ),
    ]
    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buffer.getvalue()
