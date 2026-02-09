import numpy as np
import pandas as pd
import re
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier


def rgba(hex_color, a=0.22):
    h = str(hex_color).lstrip('#')
    rgb = [int(h[i:i+2], 16) for i in (0, 2, 4)]
    return f'rgba({rgb[0]},{rgb[1]},{rgb[2]},{a})'


def wrap_html(text, max_len=30):
    s = str(text)
    if len(s) <= max_len:
        return s
    parts = s.split(' ')
    out, line = [], []
    for w in parts:
        if sum(len(x) for x in line) + len(line) + len(w) > max_len:
            out.append(' '.join(line))
            line = [w]
        else:
            line.append(w)
    if line:
        out.append(' '.join(line))
    return '<br>'.join(out)


def nice_title_case(s):
    s = str(s).replace('_', ' ').strip()
    if s.lower() == 'ir':
        return 'IR'
    return ' '.join(w[:1].upper() + w[1:] if w else w for w in s.split())


def _norm_token(s):
    return str(s).strip().lower().replace('-', '_').replace(' ', '_')

def _human_frequency(token):
    """
    Convert a one-hot suffix into a readable phrase.
    Works for things like:
      'once_or_twice_a_week' -> 'once or twice a week'
      'less_than_once_a_month' -> 'less than once a month'
      'i_dont' / 'dont' / 'never' -> 'not at all'
    """
    t = _norm_token(token)

    t = t.replace('once_or_twice', 'once_or_twice')
    t = t.replace('one_or_two', 'once_or_twice')
    t = t.replace('less_than_once_a_month', 'less_than_once_a_month')
    t = t.replace('less_than_once_month', 'less_than_once_a_month')
    t = t.replace('less_than_once_per_month', 'less_than_once_a_month')

    none_like = {'i_dont', 'dont', 'do_not', 'no', 'none', 'never', 'not_at_all'}
    if t in none_like:
        return 'not at all'

    return t.replace('_', ' ')

def make_exclude_list(exclude=None, default_exclude=None):
    exclude = list(exclude or [])
    default_exclude = list(default_exclude or [])
    return list(dict.fromkeys(default_exclude + exclude))

def pretty_question(feature_name):
    f = str(feature_name)
    m = re.match(r'^(sex_frequency)_(.+)$', f)
    if m:
        phrase = _human_frequency(m.group(2))
        if phrase == 'not at all':
            return "Do you not have sex at all?"
        return f"Do you have sex {phrase}?"

    m = re.match(r'^(masturbation_frequency)_(.+)$', f)
    if m:
        phrase = _human_frequency(m.group(2))
        if phrase == 'not at all':
            return "Do you not masturbate at all?"
        return f"Do you masturbate {phrase}?"

    m = re.match(r'^(body_count)_(.+)$', f)
    if m:
        token = _norm_token(m.group(2))
        zero_like = {'0', 'zero', 'none', 'no_one', 'noone'}
        if token in zero_like:
            return "Are you a virgin?"
        return f"Is your body count {m.group(2)}?"

    if f.startswith('porn_genres_watched__'):
        item = nice_title_case(f.replace('porn_genres_watched__', '').replace('_', ' '))
        return f"Do you watch {item} porn?"

    if f.startswith('dating_apps_used__'):
        item = nice_title_case(f.replace('dating_apps_used__', '').replace('_', ' '))
        return f"Do you use {item}?"

    if f.startswith('kinks_participated__'):
        item = nice_title_case(f.replace('kinks_participated__', '').replace('_', ' '))
        return f"Are you into {item}?"

    if f.startswith('soc__'):
        item = nice_title_case(f.replace('soc__', '').replace('_', ' '))
        return f"Are you in {item}?"

    BOOL_QUESTIONS = {
        'watches_porn': "Do you watch porn?",
        'had_threesome': "Have you had a threesome?",
        'had_same_gender_sex': "Have you had gay sex?",
        'had_sex_on_campus': "Have you had sex on campus?",
        'has_cheated': "Have you cheated?",
        'hooked_up_with_sway': "Have you hooked up with someone at/from Sway?",
        'had_std': "Have you had an STD?"
    }
    if f in BOOL_QUESTIONS:
        return BOOL_QUESTIONS[f]

    return nice_title_case(f).rstrip('?') + "?"


def bucket(p):
    if p >= 0.85:
        return "almost"
    if p >= 0.65:
        return "prob"
    if p >= 0.35:
        return "either"
    if p >= 0.15:
        return "unlikely"
    return "almost_not"


def get_leaf_styles():
    return {
        'virgin': {
            "almost": "You are almost definitely a virgin",
            "prob": "You are probably a virgin",
            "either": "Could go either way",
            "unlikely": "You are probably not a virgin",
            "almost_not": "You are almost definitely not a virgin"
        },
        'taken': {
            "almost": "You are almost definitely taken",
            "prob": "You are probably taken",
            "either": "Could go either way",
            "unlikely": "You are probably not taken",
            "almost_not": "You are almost definitely not taken"
        },
        'cheated': {
            "almost": "You have almost definitely been cheated on",
            "prob": "You have probably been cheated on",
            "either": "Could go either way",
            "unlikely": "You have probably not been cheated on",
            "almost_not": "You have almost definitely not been cheated on"
        }
    }


def leaf_text_binary(p, style):
    return style[bucket(p)]


def leaf_text_bodycount(class_probs, classes):
    i = int(np.argmax(class_probs))
    label = str(classes[i])
    conf = float(class_probs[i])
    return f"Most likely: {label}", f"{conf*100:.0f}% confidence"


def collapse_set_for_binary(tree):
    collapse = set()
    
    def node_bucket(node_id):
        l = tree.children_left[node_id]
        r = tree.children_right[node_id]
        if l == r:
            counts = tree.value[node_id][0]
            p1 = counts[1] / counts.sum() if counts.sum() else 0.0

            return bucket(p1)
        b_l = node_bucket(l)
        b_r = node_bucket(r)
        return b_l if b_l == b_r else "mix"
    
    def rec(node_id):
        l = tree.children_left[node_id]
        r = tree.children_right[node_id]
        if l == r:
            return node_bucket(node_id)
        a = rec(l)
        b = rec(r)
        if a == b and a != "mix":
            collapse.add(node_id)
            return a
        return "mix"
    
    rec(0)
    return collapse


def display_nodes(tree, collapse_nodes):
    keep = set()
    edges = []
    stack = [0]
    
    while stack:
        nid = stack.pop()
        keep.add(nid)
        
        if nid in collapse_nodes:
            continue
        
        l = tree.children_left[nid]
        r = tree.children_right[nid]
        if l == r:
            continue
        
        edges.append((nid, l, "No"))
        edges.append((nid, r, "Yes"))
        stack.append(l)
        stack.append(r)
    
    return keep, edges


def build_tree_layout(keep_nodes, edges, tree, collapse_nodes, x_span=(0.03, 0.97), level_gap=1.25,leaf_gap=1.35):    
    depth = {0: 0}
    children = {}
    for src, dst, _ in edges:
        children.setdefault(src, []).append(dst)
    
    stack = [0]
    while stack:
        nid = stack.pop()
        for ch in children.get(nid, []):
            depth[ch] = depth[nid] + 1
            stack.append(ch)
    
    max_d = max(depth.values()) if depth else 1
    
    display_leaves = []
    for nid in keep_nodes:
        l = tree.children_left[nid]
        r = tree.children_right[nid]
        if nid in collapse_nodes or l == r:
            display_leaves.append(nid)
    
    x = {nid: 0.0 for nid in keep_nodes}
    leaves_sorted = sorted(display_leaves)
    for i, nid in enumerate(leaves_sorted):
        x[nid] = float(i) * float(leaf_gap)
    
    order = sorted(list(keep_nodes), key=lambda z: -depth.get(z, 0))
    for nid in order:
        if nid in display_leaves:
            continue
        chs = children.get(nid, [])
        if len(chs) == 2:
            x[nid] = (x[chs[0]] + x[chs[1]]) / 2.0
    
    if len(display_leaves) > 1:
        xs = np.array([x[n] for n in keep_nodes], dtype=float)
        mn, mx = xs.min(), xs.max()
        for nid in keep_nodes:
            x[nid] = (x[nid] - mn) / (mx - mn) if mx > mn else 0.5
    else:
        for nid in keep_nodes:
            x[nid] = 0.5
    
    lo, hi = x_span
    for nid in keep_nodes:
        x[nid] = float(lo) + (float(hi) - float(lo)) * x[nid]
    
    den = max(1.0, (max_d * float(level_gap)) + 0.5)
    y = {nid: 1.0 - ((depth.get(nid, 0) * float(level_gap)) / den) for nid in keep_nodes}

    
    return x, y, display_leaves


def tree_to_figure_binary(clf, feature_names, title, leaf_style, theme_colors, typography, palette, height=800):
    t = clf.tree_
    collapse_nodes = collapse_set_for_binary(t)
    keep_nodes, edges = display_nodes(t, collapse_nodes)
    x, y, display_leaves = build_tree_layout(keep_nodes, edges, t, collapse_nodes,x_span=(0.02, 0.98), level_gap=1.35,leaf_gap=1.55)
    
    edge_x, edge_y = [], []
    for src, dst, _ in edges:
        edge_x += [x[src], x[dst], None]
        edge_y += [y[src], y[dst], None]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=3.2, color='rgba(43,45,66,0.20)'),
        hoverinfo='skip',
        showlegend=False
    ))
    
    leaf_color_map = {}
    for i, nid in enumerate(sorted(display_leaves, key=lambda z: x[z])):
        leaf_color_map[nid] = rgba(palette[i % len(palette)], a=0.26)
    
    node_x = [x[nid] for nid in keep_nodes]
    node_y = [y[nid] for nid in keep_nodes]
    node_sizes = [30 if nid in display_leaves else 22 for nid in keep_nodes]
    node_colors = [
        leaf_color_map.get(nid, rgba(palette[0], a=0.26)) 
        if nid in display_leaves else 'rgba(43,45,66,0.07)' 
        for nid in keep_nodes
    ]
    
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        marker=dict(size=node_sizes, color=node_colors, line=dict(width=2, color='white')),
        hoverinfo='skip',
        showlegend=False
    ))
    
    for nid in keep_nodes:
        is_leaf = nid in display_leaves
        if is_leaf:
            counts = t.value[nid][0]
            p1 = counts[1] / counts.sum() if counts.sum() else 0.0
            main = wrap_html(leaf_text_binary(p1, leaf_style), max_len=28)
            sub = f"{p1*100:.0f}% chance"
            bg = leaf_color_map.get(nid, rgba(palette[0], a=0.26))
        else:
            f_idx = t.feature[nid]
            main = wrap_html(pretty_question(feature_names[f_idx]), max_len=34)
            sub = ""
            bg = 'rgba(255,255,255,0.96)'
        
        txt = f"<b>{main}</b>" + (f"<br><span style='font-size:12px; opacity:0.82'>{sub}</span>" if sub else "")
        
        fig.add_annotation(
            x=float(x[nid]),
            y=float(y[nid]),
            text=txt,
            showarrow=False,
            align='center',
            xanchor='center',
            yanchor='middle',
            bgcolor=bg,
            bordercolor='rgba(43,45,66,0.16)',
            borderwidth=2,
            borderpad=7,
            font=dict(size=13, family=typography['font_family'], color=theme_colors['neutral_dark'])
        )
    
    for src, dst, lab in edges:
        mx = (x[src] + x[dst]) / 2.0
        my = (y[src] + y[dst]) / 2.0
        fig.add_annotation(
            x=float(mx), y=float(my),
            text=f"<b>{lab}</b>",
            showarrow=False,
            font=dict(size=12, family=typography['font_family'], color=theme_colors['neutral_dark']),
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor='rgba(43,45,66,0.10)',
            borderwidth=1,
            borderpad=3
        )
    
    fig.update_layout(
        template='plotly_white',
        height=height,
        width = 1200,
        margin=dict(l=30, r=30, t=75, b=20),
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5, xanchor='center',
            font=dict(size=typography['title_size'], family=typography['font_family'], 
                     color=theme_colors['neutral_dark'])
        ),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor='white',
        font=dict(family=typography['font_family'])
    )
    return fig


def build_feature_frame(d, base_cols, allow_multihot=True):
    MULTIHOT_PREFIXES = [
        'sex_positions_tried__', 'kinks_participated__', 'porn_genres_watched__',
        'dating_apps_used__', 'soc__', 'std__'
    ]
    
    def is_multihot(col):
        return any(str(col).startswith(p) for p in MULTIHOT_PREFIXES)
    
    use_cols = [c for c in base_cols if c in d.columns]
    
    if allow_multihot:
        extra = [c for c in d.columns if is_multihot(c)]
        use_cols = list(dict.fromkeys(use_cols + extra))
    
    X_raw = d[use_cols].copy()
    
    yesno_like = [
        'had_sex_on_campus', 'had_same_gender_sex', 'watches_porn',
        'has_cheated', 'has_been_cheated_on', 'had_threesome',
        'hooked_up_with_sway', 'had_std'
    ]
    for c in yesno_like:
        if c in X_raw.columns:
            s = X_raw[c].astype(str)
            X_raw[c] = pd.Series(
                np.where(s.eq('Yes'), 1, np.where(s.eq('No'), 0, np.nan)),
                index=X_raw.index
            )
    
    for c in X_raw.columns:
        if is_multihot(c):
            X_raw[c] = pd.to_numeric(X_raw[c], errors='coerce').fillna(0).astype(int)
    
    cat_cols = []
    for c in X_raw.columns:
        if not is_multihot(c) and not pd.api.types.is_numeric_dtype(X_raw[c]):
            cat_cols.append(c)
    
    X = pd.get_dummies(X_raw, columns=cat_cols, dummy_na=False)
    nunique = X.nunique(dropna=False)
    X = X[nunique[nunique > 1].index].copy()
    
    return X.fillna(0)


def drop_leakage_columns(X, exclude_cols):
    exclude_cols = set(exclude_cols or [])
    if not exclude_cols:
        return X
    
    drop_like = []
    for c in X.columns:
        for ex in exclude_cols:
            ex = str(ex)
            if c == ex or c.startswith(ex + '_') or c.startswith(ex + '__'):
                drop_like.append(c)
                break
    
    return X.drop(columns=sorted(set(drop_like)), errors='ignore')


def train_binary_tree(df, y_series, base_cols, exclude_cols, params):
    d = df.copy()
    d['_y_'] = y_series
    d = d.dropna(subset=['_y_']).copy()
    d['_y_'] = d['_y_'].astype(int)
    
    base_cols = [c for c in base_cols if c in d.columns and c not in set(exclude_cols or [])]
    X = build_feature_frame(d, base_cols, allow_multihot=True)
    X = drop_leakage_columns(X, exclude_cols)
    
    nunique = X.nunique(dropna=False)
    X = X[nunique[nunique > 1].index].copy()
    
    if X.shape[1] < 2 or len(X) < (params['min_samples_leaf'] * 2):
        return None, None
    
    clf = DecisionTreeClassifier(
        max_depth=params['max_depth'],
        min_samples_leaf=params['min_samples_leaf'],
        min_samples_split=params['min_samples_split'],
        random_state=params['random_state'],
        class_weight='balanced'
    )
    clf.fit(X, d['_y_'])
    
    return clf, list(X.columns)
