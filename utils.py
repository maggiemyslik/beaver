import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier


def create_sankey_diagram(df, theme_colors, typography, palette, min_n=10):
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
        x=cheat_df['been_cheated_on'],
        name='Been Cheated On',
        orientation='h',
        marker=dict(color=palette[1]),
        text=(-cheat_df['been_cheated_on']).round(1).astype(str) + '%',
        textposition='outside'
    ))

    fig.add_trace(go.Bar(
        y=cheat_df['group'],
        x=cheat_df['cheated'],
        name='Have Cheated',
        orientation='h',
        marker=dict(color=palette[0]),
        text=cheat_df['cheated'].round(1).astype(str) + '%',
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

GROUP_VARS = {
    'accommodation': 'lse_accommodation',
    'department': 'department',
    'society': 'society_or_sports_team',  
    'gender_sexuality': None
}

YES_NO_VARS = {
    'hooked_up_with_sway': 'hooked_up_with_sway',
    'cheated': 'has_cheated',
    'cheated_on': 'has_been_cheated_on',
    'sex_on_campus': 'had_sex_on_campus',
    'porn': 'watches_porn',
    'same_gender_sex': 'had_same_gender_sex',
    'threesome': 'had_threesome',
    'had_std': 'had_std',
    'std_chlamydia': 'std__chlamydia',
    'std_gonorrhea': 'std__gonorrhea',
    'std_herpes': 'std__herpes',
    'std_hpv': 'std__hpv',
    'std_hiv_aids': 'std__hiv_aids',
    'std_other': 'std__other'
}

SINGLE_CHOICE_VARS = {
    'favourite_position': 'favourite_sex_position',
    'body_count': 'body_count',
    'masturbation_frequency': 'masturbation_frequency',
    'sex_frequency': 'sex_frequency',
    'regular_partners': 'number_of_regular_sexual_partners'
}

MULTI_SELECT_PREFIXES = {
    'positions_tried': 'sex_positions_tried__',
    'kinks': 'kinks_participated__',
    'porn_genres': 'porn_genres_watched__',
    'dating_apps': 'dating_apps_used__'
}


RELATIONSHIP_COL = 'relationship_status'

REL_STATUS_MAP = {
    'taken': 'Yes',
    'single': 'No',
    'situationship': "It's complicated"
}


def relationship_rate(df, group_col, status_value, min_n):
    out = (
        df.groupby(group_col)
        .filter(lambda x: len(x) >= min_n)
        .groupby(group_col)[RELATIONSHIP_COL]
        .apply(lambda x: (x == status_value).mean())
        .sort_values(ascending=False)
    )
    return out


def queer_rate(df, group_col, min_n):
    mask = df['sexuality'].str.lower().str.contains('gay|bi', na=False)
    out = (
        df.assign(is_queer=mask)
        .groupby(group_col)
        .filter(lambda x: len(x) >= min_n)
        .groupby(group_col)['is_queer']
        .mean()
        .sort_values(ascending=False)
    )
    return out


def yes_rate(df, group_col, var, min_n):
    out = (
        df.groupby(group_col)
        .filter(lambda x: len(x) >= min_n)
        .groupby(group_col)[var]
        .apply(lambda x: (x == 'Yes').mean())
        .sort_values(ascending=False)
    )
    return out


def modal_category(df, group_col, var, min_n):
    out = (
        df.groupby(group_col)
        .filter(lambda x: len(x) >= min_n)
        .groupby(group_col)[var]
        .agg(lambda x: x.value_counts().idxmax())
    )
    return out


def multiselect_summary(df, group_col, prefix, min_n):
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        return None

    summaries = {}
    for group, sub in df.groupby(group_col):
        if len(sub) < min_n:
            continue
        counts = sub[cols].sum().sort_values(ascending=False)
        top_item = counts.index[0].replace(prefix, '').title()
        summaries[group] = {
            'most_common': top_item,
            'total_selected': counts.sum()
        }

    return pd.DataFrame(summaries).T


def generate_summary_stats(
    df,
    soc_cols=None,
    min_group_size=5,
    top_k=3,
    out_path='results/summary_statistics.txt',
    also_print=False
):
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    f = open(out_path, 'w', encoding='utf-8')

    def p(*args, **kwargs):
        print(*args, file=f, **kwargs)
        if also_print:
            print(*args, **kwargs)

    df = df.copy()
    df['gender_sexuality'] = df['gender'] + " / " + df['sexuality']

    def _pct(x):
        if pd.isna(x):
            return "NA"
        return f"{x * 100:.1f}%"

    def _safe_mean_bool(series, is_true_value='Yes'):
        series = series.dropna()
        if series.empty:
            return float('nan')
        return (series == is_true_value).mean()

    def _print_top_bottom(series, label):
        series = series.dropna()
        if series.empty:
            p(f"   {label}: (no data)")
            return

        top_name = series.index[0]
        top_val = series.iloc[0]
        bot_name = series.index[-1]
        bot_val = series.iloc[-1]

        p(f"   {label}:")
        p(f"       Highest: {top_name} ({_pct(top_val)})")
        if len(series) > 1:
            p(f"       Lowest:  {bot_name} ({_pct(bot_val)})")

        p(f"      Top {min(top_k, len(series))}:")
        for i, (idx, val) in enumerate(series.head(top_k).items(), 1):
            p(f"        {i}. {idx}: {_pct(val)}")

    def _print_modal(modal_series, label):
        modal_series = modal_series.dropna()
        if modal_series.empty:
            p(f"   {label}: (no data)")
            return

        p(f"   {label}:")
        for grp, mode in modal_series.sort_index().items():
            p(f"      - {grp}: {mode}")

    def _print_multiselect_most_common(multi_df, label):
        if multi_df is None or multi_df.empty:
            p(f"   {label}: (no data)")
            return

        if 'most_common' not in multi_df.columns:
            p(f"   {label}: (unexpected format from multiselect_summary)")
            return

        p(f"   {label} (most common):")
        for grp, row in multi_df.sort_index().iterrows():
            p(f"      - {grp}: {row['most_common']}")


    def _print_society_section():
        if soc_cols is None or len(soc_cols) == 0:
            return

        p("\n" + "=" * 90)
        p(f"SOCIETY/SPORTS BREAKDOWN (multi-hot columns)  (min n = {min_group_size})")
        p("=" * 90)

        soc_sizes = df[soc_cols].sum().sort_values(ascending=False)
        soc_keep = soc_sizes[soc_sizes >= min_group_size].index.tolist()

        p("\nGroup sizes (top 10):")
        for i, (k, v) in enumerate(soc_sizes.head(10).items(), 1):
            p(f"  {i}. {k}: {int(v)}")

        def _soc_rate(var_col, label, expects_yes_no=True):
            rates = {}
            for soc in soc_keep:
                mask = df[soc] == 1
                sub = df.loc[mask, var_col]
                if sub.empty:
                    continue
                if expects_yes_no:
                    rates[soc] = (sub == 'Yes').mean()
                else:
                    rates[soc] = sub.mean()
            out = pd.Series(rates).sort_values(ascending=False)
            _print_top_bottom(out, label)

        p("\nA) YES/NO RATES (top/bottom)")
        _soc_rate('hooked_up_with_sway', " Hooked up with Sway", expects_yes_no=True)
        _soc_rate('had_std', " Had STD", expects_yes_no=True)
        _soc_rate('has_cheated', " Have cheated", expects_yes_no=True)
        _soc_rate('has_been_cheated_on', " Have been cheated on", expects_yes_no=True)
        _soc_rate('had_sex_on_campus', " Had sex on campus", expects_yes_no=True)
        _soc_rate('watches_porn', " Watches porn", expects_yes_no=True)
        _soc_rate('had_same_gender_sex', " Had same-gender sex", expects_yes_no=True)
        _soc_rate('had_threesome', " Had a threesome", expects_yes_no=True)

        for std_col, label in [
            ('std__chlamydia', " STD: chlamydia"),
            ('std__gonorrhea', " STD: gonorrhea"),
            ('std__herpes', " STD: herpes"),
            ('std__hpv', " STD: hpv"),
            ('std__hiv_aids', " STD: hiv_aids"),
            ('std__other', " STD: other")
        ]:
            if std_col in df.columns:
                _soc_rate(std_col, f" {label}", expects_yes_no=False)

        p("\nB) SWAY RATE AMONG NON-VIRGINS (top/bottom)")
        non_virg = df[df['is_virgin'] == 'No'].copy()
        if not non_virg.empty:
            rates = {}
            for soc in soc_keep:
                mask = non_virg[soc] == 1
                sub = non_virg.loc[mask, 'hooked_up_with_sway']
                if sub.empty:
                    continue
                rates[soc] = (sub == 'Yes').mean()
            out = pd.Series(rates).sort_values(ascending=False)
            _print_top_bottom(out, " Hooked up with Sway (non-virgins only)")
        else:
            p("   (no non-virgins in data)")

        p("\nC) FAVOURITES / MODES (per society)")
        def _soc_modal(var_col, label):
            modes = {}
            for soc in soc_keep:
                mask = df[soc] == 1
                sub = df.loc[mask, var_col].dropna()
                if len(sub) < min_group_size:
                    continue
                modes[soc] = sub.value_counts().idxmax()
            out = pd.Series(modes)
            _print_modal(out, label)

        _soc_modal('favourite_sex_position', " Favourite sex position")
        _soc_modal('body_count', " Most common body count bucket")
        _soc_modal('sex_frequency', " Most common sex frequency")
        _soc_modal('masturbation_frequency', " Most common masturbation frequency")
        _soc_modal('number_of_regular_sexual_partners', " Most common # regular sexual partners")

        p("\nD) MULTI-SELECT 'MOST COMMON' (per society)")
        def _soc_multiselect(prefix, label):
            cols = [c for c in df.columns if c.startswith(prefix)]
            if not cols:
                p(f"   {label}: (no data)")
                return
            out = {}
            for soc in soc_keep:
                sub = df.loc[df[soc] == 1, cols]
                if len(sub) < min_group_size:
                    continue
                counts = sub.sum().sort_values(ascending=False)
                if counts.empty:
                    continue
                out[soc] = counts.index[0].replace(prefix, '').title()
            out = pd.Series(out)
            _print_modal(out, f" {label} (most common)")

        _soc_multiselect('sex_positions_tried__', " Positions tried")
        _soc_multiselect('porn_genres_watched__', " Porn genres watched")
        _soc_multiselect('kinks_participated__', " Kinks participated")
        _soc_multiselect('dating_apps_used__', " Dating apps used")

    def _print_lse_wide_counts():
        p("\n" + "=" * 90)
        p("LSE-WIDE STATS")
        p("=" * 90)

        p("\n Virginity")
        p(f"   Virgin rate: {_pct(_safe_mean_bool(df['is_virgin'], 'Yes'))}")
        p(f"   Non-virgin rate: {_pct(_safe_mean_bool(df['is_virgin'], 'No'))}")

        p("\n Body Count")
        bc_mode = df['body_count'].dropna().value_counts()
        if not bc_mode.empty:
            p(f"   Most common body count bucket: {bc_mode.index[0]} ({_pct(bc_mode.iloc[0] / len(df))})")
            p("   Top buckets:")
            for i, (k, v) in enumerate(bc_mode.head(top_k).items(), 1):
                p(f"     {i}. {k}: {_pct(v / len(df))}")

        p("\n Infidelity")
        p(f"   Have cheated: {_pct(_safe_mean_bool(df['has_cheated'], 'Yes'))}")
        p(f"   Have been cheated on: {_pct(_safe_mean_bool(df['has_been_cheated_on'], 'Yes'))}")

        p("\n STD")
        p(f"   Had an STD: {_pct(_safe_mean_bool(df['had_std'], 'Yes'))}")
        for std_col, label in [
            ('std__chlamydia', 'chlamydia'),
            ('std__gonorrhea', 'gonorrhea'),
            ('std__herpes', 'herpes'),
            ('std__hpv', 'hpv'),
            ('std__hiv_aids', 'hiv_aids'),
            ('std__other', 'other')
        ]:
            if std_col in df.columns:
                p(f"   {label}: {_pct(df[std_col].mean())}")

        p("\n Sway Hookups")
        sway_all = _safe_mean_bool(df['hooked_up_with_sway'], 'Yes')
        p(f"   Hooked up with someone from Sway (overall): {_pct(sway_all)}")

        if (df['is_virgin'] == 'No').any():
            sway_non_virgins = _safe_mean_bool(df.loc[df['is_virgin'] == 'No', 'hooked_up_with_sway'], 'Yes')
            p(f"   Among non-virgins: {_pct(sway_non_virgins)}")

        p("\n Sex on Campus")
        p(f"   Had sex on campus: {_pct(_safe_mean_bool(df['had_sex_on_campus'], 'Yes'))}")

        p("\n Favourite Position")
        fav_pos = df['favourite_sex_position'].dropna().value_counts()
        if not fav_pos.empty:
            p(f"   Uni favourite position: {fav_pos.index[0]} ({_pct(fav_pos.iloc[0] / len(df))})")
            p("   Top positions:")
            for i, (k, v) in enumerate(fav_pos.head(top_k).items(), 1):
                p(f"     {i}. {k}: {_pct(v / len(df))}")

        p("\n Porn")
        p(f"   Watches porn: {_pct(_safe_mean_bool(df['watches_porn'], 'Yes'))}")

        porn_cols = [c for c in df.columns if c.startswith('porn_genres_watched__')]
        if porn_cols:
            genre_counts = df[porn_cols].sum().sort_values(ascending=False)
            if not genre_counts.empty:
                top_genre = genre_counts.index[0].replace('porn_genres_watched__', '').title()
                p(f"   Favourite porn genre (most selected): {top_genre} ({_pct(genre_counts.iloc[0] / len(df))})")
                p("   Top genres:")
                for i, (k, v) in enumerate(genre_counts.head(top_k).items(), 1):
                    g = k.replace('porn_genres_watched__', '').title()
                    p(f"     {i}. {g}: {_pct(v / len(df))}")

        p("\n Masturbation Frequency")
        mf = df['masturbation_frequency'].dropna().value_counts()
        if not mf.empty:
            p(f"   Most common: {mf.index[0]} ({_pct(mf.iloc[0] / len(df))})")

        p("\n Sex Frequency")
        sf = df['sex_frequency'].dropna().value_counts()
        if not sf.empty:
            p(f"   Most common: {sf.index[0]} ({_pct(sf.iloc[0] / len(df))})")

        p("\n Regular Sexual Partners")
        rp = df['number_of_regular_sexual_partners'].dropna().value_counts()
        if not rp.empty:
            p(f"   Most common: {rp.index[0]} ({_pct(rp.iloc[0] / len(df))})")

        p("\n Positions Tried (overall)")
        pos_cols = [c for c in df.columns if c.startswith('sex_positions_tried__')]
        if pos_cols:
            pos_counts = df[pos_cols].sum().sort_values(ascending=False)
            if not pos_counts.empty:
                top_pos = pos_counts.index[0].replace('sex_positions_tried__', '').title()
                p(f"   Most tried: {top_pos} ({_pct(pos_counts.iloc[0] / len(df))})")

        p("\n Kinks (overall)")
        kink_cols = [c for c in df.columns if c.startswith('kinks_participated__')]
        if kink_cols:
            kink_counts = df[kink_cols].sum().sort_values(ascending=False)
            if not kink_counts.empty:
                top_kink = kink_counts.index[0].replace('kinks_participated__', '').title()
                p(f"   Most common: {top_kink} ({_pct(kink_counts.iloc[0] / len(df))})")

        p("\n Dating Apps (overall)")
        app_cols = [c for c in df.columns if c.startswith('dating_apps_used__')]
        if app_cols:
            app_counts = df[app_cols].sum().sort_values(ascending=False)
            if not app_counts.empty:
                top_app = app_counts.index[0].replace('dating_apps_used__', '').title()
                p(f"   Most used: {top_app} ({_pct(app_counts.iloc[0] / len(df))})")

        p("\n Relationship Status (LSE-wide)")
        for label, val in REL_STATUS_MAP.items():
            rate = (df[RELATIONSHIP_COL] == val).mean()
            p(f"   {label.title()}: {_pct(rate)}")

    group_blocks = [
        ('ACCOMMODATION', 'lse_accommodation'),
        ('DEPARTMENT', 'department'),
        ('GENDER × SEXUALITY', 'gender_sexuality')
    ]

    try:
        p("\n" + "=" * 90)
        p("THE BEAVER’S SEX SURVEY 2026 — EVERYTHING REPORT")
        p("=" * 90)

        _print_lse_wide_counts()

        for block_title, group_col in group_blocks:
            p("\n" + "=" * 90)
            p(f"{block_title} BREAKDOWN  (min n = {min_group_size})")
            p("=" * 90)

            sizes = df[group_col].value_counts(dropna=False)
            p("\nGroup sizes (top 10):")
            for i, (k, v) in enumerate(sizes.head(10).items(), 1):
                label = str(k) if pd.notna(k) else "(missing)"
                p(f"  {i}. {label}: {v}")

            p("\nA) YES/NO RATES (top/bottom)")
            _print_top_bottom(yes_rate(df, group_col, 'hooked_up_with_sway', min_group_size), " Hooked up with Sway")
            _print_top_bottom(yes_rate(df, group_col, 'had_std', min_group_size), " Had STD")
            _print_top_bottom(yes_rate(df, group_col, 'has_cheated', min_group_size), " Have cheated")
            _print_top_bottom(yes_rate(df, group_col, 'has_been_cheated_on', min_group_size), " Have been cheated on")
            _print_top_bottom(yes_rate(df, group_col, 'had_sex_on_campus', min_group_size), " Had sex on campus")
            _print_top_bottom(yes_rate(df, group_col, 'watches_porn', min_group_size), " Watches porn")
            _print_top_bottom(yes_rate(df, group_col, 'had_same_gender_sex', min_group_size), " Had same-gender sex")
            _print_top_bottom(yes_rate(df, group_col, 'had_threesome', min_group_size), " Had a threesome")

            for std_col, label in [
                ('std__chlamydia', " STD: chlamydia"),
                ('std__gonorrhea', " STD: gonorrhea"),
                ('std__herpes', " STD: herpes"),
                ('std__hpv', " STD: hpv"),
                ('std__hiv_aids', " STD: hiv_aids"),
                ('std__other', " STD: other")
            ]:
                if std_col in df.columns:
                    rates = (
                        df.groupby(group_col)
                        .filter(lambda x: len(x) >= min_group_size)
                        .groupby(group_col)[std_col]
                        .mean()
                        .sort_values(ascending=False)
                    )
                    _print_top_bottom(rates, f" {label}")

            p("\nC) RELATIONSHIP STATUS (top/bottom)")
            _print_top_bottom(
                relationship_rate(df, group_col, REL_STATUS_MAP['taken'], min_group_size),
                " Taken (in a relationship)"
            )
            _print_top_bottom(
                relationship_rate(df, group_col, REL_STATUS_MAP['single'], min_group_size),
                " Single"
            )
            _print_top_bottom(
                relationship_rate(df, group_col, REL_STATUS_MAP['situationship'], min_group_size),
                " Situationships (It's complicated)"
            )

            p("\nD) QUEER CONCENTRATION")
            _print_top_bottom(
                queer_rate(df, group_col, min_group_size),
                " Gay / Bi concentration"
            )

            p("\nB) SWAY RATE AMONG NON-VIRGINS (top/bottom)")
            non_virg = df[df['is_virgin'] == 'No'].copy()
            if not non_virg.empty:
                _print_top_bottom(
                    yes_rate(non_virg, group_col, 'hooked_up_with_sway', min_group_size),
                    " Hooked up with Sway (non-virgins only)"
                )
            else:
                p("   (no non-virgins in data)")

            p("\nC) FAVOURITES / MODES (per group)")
            _print_modal(modal_category(df, group_col, 'favourite_sex_position', min_group_size), " Favourite sex position")
            _print_modal(modal_category(df, group_col, 'body_count', min_group_size), " Most common body count bucket")
            _print_modal(modal_category(df, group_col, 'sex_frequency', min_group_size), " Most common sex frequency")
            _print_modal(modal_category(df, group_col, 'masturbation_frequency', min_group_size), " Most common masturbation frequency")
            _print_modal(modal_category(df, group_col, 'number_of_regular_sexual_partners', min_group_size), " Most common # regular sexual partners")

            p("\nD) MULTI-SELECT 'MOST COMMON' (per group)")
            _print_multiselect_most_common(
                multiselect_summary(df, group_col, 'sex_positions_tried__', min_group_size),
                " Positions tried"
            )
            _print_multiselect_most_common(
                multiselect_summary(df, group_col, 'porn_genres_watched__', min_group_size),
                "Porn genres watched"
            )
            _print_multiselect_most_common(
                multiselect_summary(df, group_col, 'kinks_participated__', min_group_size),
                " Kinks participated"
            )
            _print_multiselect_most_common(
                multiselect_summary(df, group_col, 'dating_apps_used__', min_group_size),
                " Dating apps used"
            )

        _print_society_section()

        p("\n" + "=" * 90)
        p(f"DONE   (Wrote full 'everything' report to {out_path})")
        p("=" * 90)

    finally:
        f.close()

    return out_path