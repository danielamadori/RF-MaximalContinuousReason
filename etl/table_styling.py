from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

# Column -> colormap mapping consistent with the notebook legend
COLUMN_COLORMAPS = {
    'n_estimators': 'Reds',
    'eu_complexity': 'Oranges',
    'n_features': 'YlGn',
    'mean_eu': 'Greens',
    'mean eu features': 'Greens',
    'eu_min': 'Greens',
    'eu_max': 'Greens',
    'eu_std_dev': 'Greens',
    'eu std': 'Greens',
    'Candidate': 'Purples',
    'Reason': 'Greens',
    'Non-reason': 'Blues',
    'Candidate Anti-reason': 'Oranges',
    'Anti-reason': 'Reds',
    'Good profile': 'BuGn',
    'Bad profile': 'PuRd',
    'Preferred reason': 'PuBu',
    'Anti-reason profile': 'YlGnBu',
    'Total': 'Greys',
    'Worker span (s)': 'PuBuGn',
    'avg_depth': 'GnBu',
    'avg_leaves': 'Blues',
    'avg_nodes': 'BuPu',
    'train_size': 'Purples',
    'test_size': 'RdPu',
    'Train Size': 'Purples',
    'Test Size': 'RdPu',
    'N Estimators': 'Reds',
    'Total time (s) max': 'YlOrBr',
    'Total time (s) mean': 'YlOrBr',
    'total time': 'YlOrBr',
    'Total Time (ms)': 'YlOrBr',
    'ICF checks': 'PuBuGn',
    'ICF Checks': 'PuBuGn',
    'Reason check iteration total': 'Purples',
    'Reason check iteration': 'Purples',
    'Reason Check Iteration': 'Purples',
    'IterGoodRatio': 'Greens',
    'IterBadRatio': 'Reds',
    'IterGoodRadio %': 'Greens',
    'IterBadRadio %': 'Reds',
    'Early Stop Good total': 'PuBu',
    'Early Stop Good': 'PuBu',
    'Early Stop from Good': 'BuGn',
    'Early Stop from Bad': 'Oranges',
    'Mean EU Features': 'Greens',
    'EU Std': 'Greens',
    'Filtrered rate': 'YlGnBu',
    'filtrered rate': 'YlGnBu',
    'Filtrered Rate %': 'YlGnBu',
}
DEFAULT_CMAP = 'Greys'
CMAP_MIN_VALUE = 0.25
CMAP_COLOR_POINTS = 256

ANALYZED_COLORS = {
    'YES': 'background-color: #b6e192',
    'NO': 'background-color: #f6583e',
    'N/A': '',
}


@lru_cache(maxsize=None)
def _get_truncated_cmap(name: str) -> LinearSegmentedColormap:
    try:
        base = cm.get_cmap(name)
    except ValueError:
        base = cm.get_cmap(DEFAULT_CMAP)
    colors = base(np.linspace(CMAP_MIN_VALUE, 1.0, CMAP_COLOR_POINTS))
    return LinearSegmentedColormap.from_list(f'{name}_trimmed', colors)


def _int_formatter(value: Any) -> str:
    if pd.isna(value):
        return ''
    return f'{int(round(float(value))):d}'


def _float_formatter(decimals: int):
    def formatter(value: Any) -> str:
        if pd.isna(value):
            return ''
        return f'{float(value):.{decimals}f}'

    return formatter


def style_summary_table(df: pd.DataFrame) -> Any:
    styled = df.style

    int_columns = [col for col in df.attrs.get('format_int_columns', []) if col in df.columns]
    float_columns = [col for col in df.attrs.get('format_float_columns', []) if col in df.columns and col not in int_columns]
    default_numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if not float_columns:
        float_columns = [col for col in default_numeric_cols if col not in int_columns]
    decimals = int(df.attrs.get('format_float_decimals', 3))

    if int_columns:
        styled = styled.format({col: _int_formatter for col in int_columns})
    if float_columns:
        float_formatter = _float_formatter(decimals)
        styled = styled.format({col: float_formatter for col in float_columns})

    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        col_min = df[col].min(skipna=True)
        col_max = df[col].max(skipna=True)
        if pd.isna(col_min) or pd.isna(col_max):
            continue
        ''' TODO coloration disabled
        cmap = COLUMN_COLORMAPS.get(col, DEFAULT_CMAP)
        cmap_obj = _get_truncated_cmap(cmap)
        if col_min == col_max:
            styled = styled.background_gradient(subset=[col], cmap=cmap_obj)
        else:
            styled = styled.background_gradient(
                subset=[col],
                cmap=cmap_obj,
                vmin=col_min,
                vmax=col_max,
            )
        '''
    if 'analyzed' in df.columns:
        def analyzed_style(val: Any) -> str:
            return ANALYZED_COLORS.get(val, '')

        styled = styled.map(analyzed_style, subset=['analyzed'])
    return styled


def print_color_legend() -> None:
    header = "=" * 80
    lines = [header, "COLUMN COLOR GRADIENTS", header]
    for column, cmap in COLUMN_COLORMAPS.items():
        lines.append(f"  {column:<22} -> {cmap} (min -> max)")
    lines.append("  analyzed             -> Green (YES) / Red (NO)")
    lines.append(header)
    print("\n".join(lines))
