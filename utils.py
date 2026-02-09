"""
Advanced visualization utilities for LSE Sex Survey Analysis Part 2
Includes functions for heatmaps, sankey diagrams, and diverging charts
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier


def create_sankey_diagram(df, theme_colors, typography, palette, min_n=10):
    """Create virginity → gender → body count flow diagram"""
    BC_ORDER = ['0', '1', '2', '3-5', '5-10', '10-20', '20-50', '50-100', '100+']
    
    sankey_data = df[
        df['is_virgin'].notna() & 
        df['body_count'].notna() & 
        df['gender'].notna()
    ].copy()
    sankey_data = sankey_data[sankey_data['gender'].isin(['Men', 'Women'])]
    
    if len(sankey_data) <= min_n:
        return None
    
    flows = []
    for _, row in sankey_data.iterrows():
        virgin_status = "Virgin" if row['is_virgin'] == 'Yes' else "Non-Virgin"
        gender = row['gender']
        bc = row['body_count']
        flows.append({'virgin': virgin_status, 'gender': gender, 'body_count': bc})
    
    flow_df = pd.DataFrame(flows)
    
    nodes = []
    node_map = {}
    idx = 0
    
    for v in ['Virgin', 'Non-Virgin']:
        node_map[f"virgin_{v}"] = idx
        nodes.append(v)
        idx += 1
    
    for g in ['Men', 'Women']:
        node_map[f"gender_{g}"] = idx
        nodes.append(g)
        idx += 1
    
    for bc in BC_ORDER:
        node_map[f"body_count_{bc}"] = idx
        nodes.append(bc)
        idx += 1
    
    sources, targets, values = [], [], []
    
    for (v, g), count in flow_df.groupby(['virgin', 'gender']).size().items():
        if pd.notna(v) and pd.notna(g):
            sources.append(node_map[f"virgin_{v}"])
            targets.append(node_map[f"gender_{g}"])
            values.append(count)
    
    for (g, bc), count in flow_df.groupby(['gender', 'body_count']).size().items():
        if pd.notna(g) and pd.notna(bc) and bc in BC_ORDER:
            sources.append(node_map[f"gender_{g}"])
            targets.append(node_map[f"body_count_{bc}"])
            values.append(count)
    
    node_colors = (
        [palette[0], palette[1]] +
        [palette[2], palette[3]] +
        [palette[i % len(palette)] for i in range(len(BC_ORDER))]
    )
    
    n_virgin, n_gender, n_bc = 2, 2, len(BC_ORDER)
    
    def spaced_y(n):
        return [0.5] if n == 1 else [i/(n-1) for i in range(n)]
    
    node_x = [0.01] * n_virgin + [0.50] * n_gender + [0.99] * n_bc
    node_y = spaced_y(n_virgin) + spaced_y(n_gender) + spaced_y(n_bc)
    
    fig = go.Figure(data=[go.Sankey(
        arrangement='fixed',
        node=dict(
            pad=20,
            thickness=25,
            line=dict(color='white', width=2),
            label=nodes,
            color=node_colors,
            x=node_x,
            y=node_y
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color='rgba(0,0,0,0.15)'
        )
    )])
    
    fig.update_layout(
        title=dict(
            text="Body Count Breakdown by Gender and Virginity Status",
            font=dict(size=typography['title_size'], 
                     color=theme_colors['neutral_dark'],
                     family=typography['font_family']),
            x=0.5,
            xanchor='center'
        ),
        font=dict(size=13, family=typography['font_family']),
        height=700,
        template='plotly_white',
        margin=dict(l=20, r=20, t=80, b=20)
    )
    
    return fig


def create_infidelity_diverging(data, group_col, title, theme_colors, typography, palette, min_n=10):
    """Create diverging bar chart comparing cheating vs being cheated on"""
    subset = data[data[group_col].notna()].copy()
    subset = subset[~subset[group_col].astype(str).str.lower().isin(['nan', 'none'])].copy()
    
    group_counts = subset[group_col].value_counts()
    valid_groups = group_counts[group_counts >= min_n].index
    subset = subset[subset[group_col].isin(valid_groups)]
    
    if subset.empty:
        return None
    
    cheat_summary = []
    for grp in valid_groups:
        grp_data = subset[subset[group_col] == grp]
        cheated = (grp_data['has_cheated'] == 'Yes').mean() * 100
        been_cheated = (grp_data['has_been_cheated_on'] == 'Yes').mean() * 100
        cheat_summary.append({
            'group': grp,
            'cheated': cheated,
            'been_cheated_on': -been_cheated
        })
    
    cheat_df = pd.DataFrame(cheat_summary).sort_values('cheated', ascending=True)
    
    if cheat_df.empty:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=cheat_df['group'],
        x=cheat_df['cheated'],
        name='Have Cheated',
        orientation='h',
        marker=dict(color=palette[0]),
        text=cheat_df['cheated'].round(1).astype(str) + '%',
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        y=cheat_df['group'],
        x=cheat_df['been_cheated_on'],
        name='Been Cheated On',
        orientation='h',
        marker=dict(color=palette[1]),
        text=(-cheat_df['been_cheated_on']).round(1).astype(str) + '%',
        textposition='outside'
    ))
    
    max_val = max(cheat_df['cheated'].max(), -cheat_df['been_cheated_on'].min())
    range_val = int(np.ceil(max_val / 10) * 10) + 10
    
    fig.update_layout(
        title=dict(
            text=f"{title}<br><sub>Who cheats vs who gets cheated on</sub>",
            font=dict(size=typography['title_size'], 
                     color=theme_colors['neutral_dark'],
                     family=typography['font_family']),
            x=0.5,
            xanchor='center'
        ),
        barmode='overlay',
        xaxis=dict(
            title='',
            range=[-range_val, range_val],
            tickvals=[-40, -20, 0, 20, 40],
            ticktext=['40%', '20%', '0%', '20%', '40%']
        ),
        yaxis=dict(title=''),
        template='plotly_white',
        height=500,
        showlegend=True,
        legend=dict(
            x=0.5,
            y=-0.15,
            xanchor='center',
            orientation='h'
        ),
        margin=dict(l=100, r=50, t=90, b=80),
        font=dict(family=typography['font_family'])
    )
    
    return fig


def create_heatmap(df, theme_colors, typography, palette):
    """Create body count × sex frequency heatmap"""
    if 'body_count' not in df.columns or 'sex_frequency' not in df.columns:
        return None
    
    heatmap_data = df[df['body_count'].notna() & df['sex_frequency'].notna()].copy()
    
    BC_ORDER = ['0', '1', '2', '3-5', '5-10', '10-20', '20-50', '50-100', '100+']
    FREQ_ORDER = ['Less', 'Once a month', 'Once or twice a week', 
                  'More than five days a week', 'Once a day', 'Multiple times a day']
    
    matrix = pd.crosstab(
        heatmap_data['sex_frequency'],
        heatmap_data['body_count'],
        normalize='all'
    ) * 100
    
    matrix = matrix.reindex(index=FREQ_ORDER, columns=BC_ORDER, fill_value=0)
    
    custom_colorscale = [
        [0.0, theme_colors['neutral_light']],
        [0.5, palette[2]],
        [1.0, palette[0]]
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix.values,
        x=matrix.columns,
        y=matrix.index,
        colorscale=custom_colorscale,
        text=matrix.values.round(1),
        texttemplate='<b>%{text}%</b>',
        textfont={"size": 12, "color": "white"},
        colorbar=dict(
            title=dict(text="% of Students", side='right'),
            thickness=20,
            len=0.7
        )
    ))
    
    fig.update_layout(
        title=dict(
            text="<b>Experience vs Activity</b><br><sub>Body count compared to current sexual frequency</sub>",
            font=dict(size=typography['title_size'], 
                     color=theme_colors['neutral_dark'],
                     family=typography['font_family']),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(title='<b>Body Count</b>', tickangle=-45, side='bottom',
                  tickfont=dict(family=typography['font_family'])),
        yaxis=dict(title='<b>Sex Frequency</b>',
                  tickfont=dict(family=typography['font_family'])),
        template='plotly_white',
        height=550,
        width=950,
        margin=dict(l=150, r=100, t=100, b=100),
        font=dict(family=typography['font_family'])
    )
    
    return fig


def create_body_count_facet(df, facet_col, title, theme_colors, typography, palette, min_n=10):
    """Create faceted body count distribution"""
    BC_ORDER = ['0', '1', '2', '3-5', '5-10', '10-20', '20-50', '50-100', '100+']
    
    if 'body_count' not in df.columns or facet_col not in df.columns:
        return None
    
    sub = df.dropna(subset=[facet_col, 'body_count']).copy()
    sub = sub[~sub[facet_col].astype(str).str.lower().isin(['nan', 'none'])].copy()
    
    if facet_col == 'sexuality':
        sub = sub[sub['sexuality'].isin(['Straight', 'Bisexual', 'Gay'])]
    
    sizes = sub[facet_col].value_counts()
    keep = sizes[sizes >= min_n].index
    sub = sub[sub[facet_col].isin(keep)]
    
    if sub.empty:
        return None
    
    sub['body_count'] = pd.Categorical(sub['body_count'], categories=BC_ORDER, ordered=True)
    
    counts = (
        sub.groupby([facet_col, 'body_count'], observed=True)
        .size()
        .reset_index(name='n')
    )
    counts['pct'] = counts.groupby(facet_col)['n'].transform(lambda x: x / x.sum() * 100)
    
    facet_orders = ['Straight', 'Bisexual', 'Gay'] if facet_col == 'sexuality' else None
    
    fig = px.bar(
        counts,
        x='body_count',
        y='pct',
        facet_col=facet_col,
        facet_col_wrap=3,
        category_orders=(
            {'body_count': BC_ORDER, facet_col: facet_orders}
            if facet_orders else {'body_count': BC_ORDER}
        ),
        color=facet_col,
        color_discrete_sequence=palette
    )
    
    fig.update_yaxes(title_text='')
    fig.update_xaxes(title='', tickangle=-35)
    fig.update_layout(
        yaxis=dict(title='Percent'),
        template='plotly_white',
        height=1000 if facet_col == 'department' else 800,
        showlegend=False,
        font=dict(family=typography['font_family']),
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=typography['title_size'], 
                     color=theme_colors['neutral_dark'],
                     family=typography['font_family']),
            x=0.5,
            xanchor='center'
        )
    )
    
    for annotation in fig.layout.annotations:
        if '=' in str(annotation.text):
            annotation.text = annotation.text.split('=')[1]
            annotation.font.size = 18
    
    return fig


def create_frequency_facet(df, facet_col, cat_col, cat_order, title, theme_colors, typography, palette, min_n=10):
    """Create faceted frequency distribution"""
    sub = df.dropna(subset=[facet_col, cat_col]).copy()
    
    if facet_col == 'sexuality':
        sub = sub[sub['sexuality'].isin(['Straight', 'Bisexual', 'Gay'])]
    
    sizes = sub[facet_col].value_counts()
    keep = sizes[sizes >= min_n].index
    sub = sub[sub[facet_col].isin(keep)]
    
    if sub.empty:
        return None
    
    sub[cat_col] = pd.Categorical(sub[cat_col], categories=cat_order, ordered=True)
    
    counts = (
        sub.groupby([facet_col, cat_col], observed=True)
        .size()
        .reset_index(name='n')
    )
    counts['pct'] = counts.groupby(facet_col)['n'].transform(lambda x: x / x.sum() * 100)
    
    facet_orders = ['Straight', 'Bisexual', 'Gay'] if facet_col == 'sexuality' else None
    
    fig = px.bar(
        counts,
        x=cat_col,
        y='pct',
        facet_col=facet_col,
        facet_col_wrap=3,
        category_orders=(
            {cat_col: cat_order, facet_col: facet_orders}
            if facet_orders else {cat_col: cat_order}
        ),
        color=cat_col,
        color_discrete_sequence=palette
    )
    
    fig.update_yaxes(title_text='')
    fig.update_xaxes(title='', showticklabels=False)
    fig.update_layout(
        yaxis=dict(title='Percent'),
        template='plotly_white',
        height=800,
        showlegend=True,
        legend=dict(
            orientation='v',
            yanchor='top',
            y=1.0,
            xanchor='left',
            x=1.02
        ),
        font=dict(family=typography['font_family']),
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=typography['title_size'], 
                     color=theme_colors['neutral_dark'],
                     family=typography['font_family']),
            x=0.5,
            xanchor='center'
        )
    )
    
    for annotation in fig.layout.annotations:
        if '=' in str(annotation.text):
            annotation.text = annotation.text.split('=')[1]
            annotation.font.size = 18
    
    return fig
