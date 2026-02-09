# LSE Sex Survey 2026 Analysis

Complete analysis pipeline for The Beaver's annual sex survey data.

## Files

1. **`LSE_Sex_Survey_Analysis.ipynb`** - Main Jupyter notebook containing all analysis code
2. **`tree_utils.py`** - Utility functions for decision tree visualizations
3. **`survey.csv`** - Input data file (must be provided)

## Quick Start

1. Place your `survey.csv` file in the same directory
2. Open `LSE_Sex_Survey_Analysis.ipynb` in Jupyter
3. Run all cells (Kernel → Restart & Run All)

## Output Structure

Running the notebook creates a `results/` directory with organized outputs:

```
results/
├── summary_statistics/
│   └── summary_statistics.txt
├── bar_charts/
│   ├── virginity_by_department.html
│   ├── cheating_by_gender.html
│   └── ... (more bar charts)
├── faceted_charts/
│   ├── body_count_by_gender.html
│   └── ... (multi-panel charts)
├── wordclouds/
│   ├── kinks.png
│   ├── porn_genres.png
│   └── ... (word visualizations)
├── decision_trees/
│   ├── virgin_prediction.html
│   ├── body_count_prediction.html
│   └── ... (interactive trees)
├── flow_diagrams/
│   └── sankey_virginity_to_experience.html
└── diverging_charts/
    ├── infidelity_gap_department.html
    └── ... (comparison charts)
```

## Customization

### Theme Colors

Edit the `THEME_COLORS` dictionary in the first code cell:

```python
THEME_COLORS = {
    'valentine': ['#C9184A', '#FF4D6D', '#FF758F', '#FF8FA3', '#FFCCD5', '#FFC2D1'],
    'accent': ['#FF006E', '#FB5607', '#FFBE0B', '#8338EC', '#3A86FF'],
    ...
}
```

**Current theme:** Valentine's Day (pinks and reds)  
**To change:** Replace hex color codes with your preferred palette

### Typography

Edit the `TYPOGRAPHY` dictionary:

```python
TYPOGRAPHY = {
    'font_family': 'Georgia, Times New Roman, serif',  # Change font here
    'title_size': 24,
    'subtitle_size': 14,
    'label_size': 13
}
```

**Current font:** Georgia/Times New Roman (serif)  
**Alternatives:** 
- `'Arial, Helvetica, sans-serif'` for sans-serif
- `'Courier New, monospace'` for monospace
- Any web-safe font family

### Analysis Parameters

Edit `ANALYSIS_PARAMS` to control:

- `min_group_size`: Minimum respondents per group to display (default: 5)
- `max_groups_display`: Maximum groups shown in charts (default: 20)
- `tree_max_depth_short`: Complexity of simple decision trees (default: 5)
- `tree_max_depth_long`: Complexity of body count trees (default: 8)
- `wordcloud_min_tokens`: Minimum words needed for wordclouds (default: 40)

## Troubleshooting

### Decision Tree Labels Overlapping

If decision tree questions overlap, try:
1. Increase `height` parameter in tree visualization calls
2. Reduce `max_depth` in `ANALYSIS_PARAMS`
3. Edit `pretty_question()` function in `tree_utils.py` to shorten specific questions

### Missing Visualizations

Check that `survey.csv` has required columns:
- `Q1` through `Q29` (question responses)
- At least 50+ valid responses after cleaning

### Font Not Displaying

Web-safe fonts work in all browsers. If using custom fonts:
1. Add `@import` statement to notebook
2. Or stick with: Georgia, Arial, Times New Roman, Courier New

## Output File Formats

- **HTML files**: Interactive charts (open in browser, hover for details)
- **PNG files**: Static wordcloud images
- **TXT file**: Summary statistics report

All HTML charts can be opened directly in a web browser and are self-contained (no server needed).

## Requirements

- pandas
- numpy
- plotly
- matplotlib
- seaborn
- wordcloud
- scikit-learn

Install with: `pip install pandas numpy plotly matplotlib seaborn wordcloud scikit-learn`
