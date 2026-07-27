def _null_summary(analysis):
    valid_count = analysis['valid_count']
    null_count = analysis['null_count']
    total_count = valid_count + null_count
    if not null_count:
        return (
            f'Se analizaron {valid_count} registros válidos y no se detectaron '
            'valores nulos.'
        )

    null_percentage = round(null_count * 100 / total_count, 2)
    return (
        f'Se analizaron {valid_count} registros válidos y se identificaron '
        f'{null_count} valores nulos, equivalentes al {null_percentage}% '
        'del total.'
    )


def _quantitative_interpretation(analysis):
    metrics = analysis['metrics']
    mean = float(metrics['mean'])
    median = float(metrics['median'])
    standard_deviation = float(metrics['standard_deviation'])
    observed_range = float(metrics['range'])
    paragraphs = [
        (
            f'La columna “{analysis["column"]}” es cuantitativa. '
            f'{_null_summary(analysis)}'
        )
    ]

    if observed_range == 0:
        paragraphs.append(
            f'Todos los registros válidos tienen el mismo valor '
            f'({_display(metrics["minimum"])}), por lo que no existe '
            'variabilidad en los datos.'
        )
    else:
        center_difference = abs(mean - median)
        comparison_scale = standard_deviation or observed_range
        if center_difference <= comparison_scale * 0.25:
            center_summary = (
                'La media y la mediana son cercanas, por lo que el centro de '
                'los datos no presenta un desplazamiento marcado.'
            )
        elif mean > median:
            center_summary = (
                'La media es mayor que la mediana, lo que sugiere una posible '
                'influencia de valores altos sobre el promedio.'
            )
        else:
            center_summary = (
                'La media es menor que la mediana, lo que sugiere una posible '
                'influencia de valores bajos sobre el promedio.'
            )

        dispersion_ratio = standard_deviation / observed_range
        if dispersion_ratio <= 0.15:
            dispersion_level = 'baja'
        elif dispersion_ratio <= 0.30:
            dispersion_level = 'moderada'
        else:
            dispersion_level = 'alta'
        paragraphs.append(
            f'La media es {_display(metrics["mean"])} y la mediana es '
            f'{_display(metrics["median"])}. {center_summary} La desviación '
            f'estándar es {_display(metrics["standard_deviation"])}, que '
            f'representa una dispersión {dispersion_level} en relación con '
            'el rango observado.'
        )
        paragraphs.append(
            f'Los valores se encuentran entre {_display(metrics["minimum"])} '
            f'y {_display(metrics["maximum"])}, con un rango de '
            f'{_display(metrics["range"])}.'
        )

    mode = metrics['mode']
    if mode == 'Sin moda':
        paragraphs.append(
            'No se identificó una moda porque ningún valor se repite con '
            'mayor frecuencia que los demás.'
        )
    else:
        paragraphs.append(
            f'El valor o conjunto de valores con mayor frecuencia es '
            f'{mode}. Esta medida señala la respuesta más recurrente, pero '
            'no implica por sí sola que represente a la mayoría de los datos.'
        )
    return paragraphs


def _qualitative_interpretation(analysis):
    modes = analysis['modes']
    mode_frequency = analysis['mode_frequency']
    mode_percentage = analysis['mode_percentage']
    paragraphs = [
        (
            f'La columna “{analysis["column"]}” es cualitativa. '
            f'{_null_summary(analysis)} Se identificaron '
            f'{analysis["unique_count"]} categorías distintas.'
        )
    ]

    if analysis['unique_count'] == 1:
        paragraphs.append(
            f'Todos los registros válidos pertenecen a la categoría '
            f'“{modes[0]}”, por lo que no existe diversidad entre las '
            'respuestas observadas.'
        )
        return paragraphs

    if len(modes) == 1:
        paragraphs.append(
            f'La categoría con mayor frecuencia es “{modes[0]}”, con '
            f'{mode_frequency} registros ({mode_percentage}% del total válido).'
        )
    else:
        mode_list = '”, “'.join(modes)
        paragraphs.append(
            f'Las categorías “{mode_list}” comparten la frecuencia más alta, '
            f'con {mode_frequency} registros cada una '
            f'({mode_percentage}% del total válido por categoría). Por ello, '
            'la distribución es multimodal.'
        )

    if mode_percentage >= 50:
        concentration = (
            'La categoría predominante concentra al menos la mitad de los '
            'registros, por lo que existe una concentración marcada.'
        )
    elif mode_percentage >= 30:
        concentration = (
            'Existe una categoría predominante, aunque no representa a la '
            'mayoría absoluta de los registros.'
        )
    else:
        concentration = (
            'Ninguna categoría concentra por sí sola una proporción elevada, '
            'por lo que las respuestas se encuentran distribuidas entre '
            'diferentes categorías.'
        )
    paragraphs.append(concentration)
    return paragraphs


def _display(value):
    return str(value)


def build_statistical_interpretation(analysis):
    """Describe statistical patterns without assuming domain-specific meaning."""
    if analysis['kind'] == 'quantitative':
        return _quantitative_interpretation(analysis)
    return _qualitative_interpretation(analysis)
