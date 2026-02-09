import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, chisquare
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD DATA
# ============================================================================

survey = pd.read_csv('survey_cleaned.csv')
print(f"Survey sample size: {len(survey)}")
print("\n" + "="*80)

# ============================================================================
# 2. KNOWN LSE DEMOGRAPHICS (2024/25)
# ============================================================================

LSE_TOTAL_STUDENTS = 12950

LSE_DEMOGRAPHICS = {
    'gender': {
        'Men': 0.43,      
        'Women': 0.57,   
    },
    
    'year_of_study': {
        'Undergraduate Year 1': 0.2,
        'Undergraduate Year 2': 0.2,
        'Undergraduate Year 3': 0.15,
        'Undergraduate Year 4': 0.05,
        'Postgraduate': 0.4,
    },
    
    'department': {
        'Economics': 0.15,
        'Management': 0.12,
        'Law': 0.10,
        'Government': 0.08,
        'International Relations': 0.08,
        'Accounting and Finance': 0.07,
        'Mathematics': 0.05,
        'Other': 0.35,  
    },
    
    'student_type': {
        'Domestic (UK)': 0.35,
        'International': 0.65,
    }
}

# ============================================================================
# 3. HELPER FUNCTIONS
# ============================================================================

def clean_gender(series):
    """Standardize gender labels"""
    s = series.astype(str).str.lower()
    s = s.replace({'male': 'Men', 'female': 'Women', 'men': 'Men', 'women': 'Women'})
    return s

def chi_square_goodness_of_fit(observed_counts, expected_proportions, category_name):
    """
    Perform chi-square goodness of fit test
    
    Returns: chi2 statistic, p-value, interpretation
    """
    total = observed_counts.sum()
    expected_counts = pd.Series({k: v * total for k, v in expected_proportions.items()})
    
    # Align categories
    common_cats = list(set(observed_counts.index) & set(expected_counts.index))
    obs = observed_counts[common_cats].values
    exp = expected_counts[common_cats].values
    
    # Chi-square test
    chi2_stat, p_value = chisquare(f_obs=obs, f_exp=exp)
    
    # Effect size (Cramér's V for goodness of fit)
    cramers_v = np.sqrt(chi2_stat / (total * (len(common_cats) - 1)))
    
    print(f"\n{'='*80}")
    print(f"CHI-SQUARE GOODNESS OF FIT TEST: {category_name.upper()}")
    print(f"{'='*80}")
    print(f"\nSample size: {total}")
    print(f"Categories compared: {len(common_cats)}")
    print(f"\nObserved vs Expected proportions:")
    
    comparison = pd.DataFrame({
        'Observed_N': observed_counts[common_cats],
        'Expected_N': expected_counts[common_cats],
        'Observed_%': (observed_counts[common_cats] / total * 100).round(1),
        'Expected_%': (expected_counts[common_cats] / total * 100).round(1),
        'Difference_%': ((observed_counts[common_cats] / total * 100) - 
                        (expected_counts[common_cats] / total * 100)).round(1)
    })
    print(comparison)
    
    print(f"\n{'─'*80}")
    print(f"Chi-square statistic: {chi2_stat:.4f}")
    print(f"P-value: {p_value:.6f}")
    print(f"Cramér's V (effect size): {cramers_v:.4f}")
    
    # Interpretation
    if p_value < 0.001:
        sig = "highly significant (p < 0.001)"
    elif p_value < 0.01:
        sig = "very significant (p < 0.01)"
    elif p_value < 0.05:
        sig = "significant (p < 0.05)"
    else:
        sig = "not significant (p ≥ 0.05)"
    
    if cramers_v < 0.1:
        effect = "negligible"
    elif cramers_v < 0.3:
        effect = "small"
    elif cramers_v < 0.5:
        effect = "medium"
    else:
        effect = "large"
    
    print(f"\nResult: The difference is {sig}")
    print(f"Effect size: {effect}")
    
    if p_value < 0.05:
        print("\n⚠️  CONCLUSION: Sample is NOT representative of LSE population on this dimension")
        print("   Consider weighting or acknowledging this limitation")
    else:
        print("\n✓  CONCLUSION: Sample appears representative of LSE population on this dimension")
    
    return chi2_stat, p_value, cramers_v, comparison

def calculate_margin_of_error(proportion, sample_size, confidence_level=0.95):
    """
    Calculate margin of error for a proportion
    
    proportion: observed proportion (0-1)
    sample_size: n
    confidence_level: typically 0.95
    """
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    se = np.sqrt(proportion * (1 - proportion) / sample_size)
    moe = z_score * se
    return moe

def sample_size_analysis(survey_n, population_n, confidence_level=0.95, margin_of_error=0.05):
    """
    Assess whether sample size is adequate
    """
    print(f"\n{'='*80}")
    print(f"SAMPLE SIZE ADEQUACY ANALYSIS")
    print(f"{'='*80}")
    
    # Required sample size (infinite population formula)
    z = stats.norm.ppf((1 + confidence_level) / 2)
    p = 0.5  # Most conservative estimate
    required_infinite = (z**2 * p * (1-p)) / (margin_of_error**2)
    
    # Finite population correction
    if population_n:
        required_finite = required_infinite / (1 + (required_infinite - 1) / population_n)
    else:
        required_finite = required_infinite
    
    print(f"\nPopulation size: {population_n:,}")
    print(f"Sample size: {survey_n:,}")
    print(f"Response rate: {(survey_n/population_n)*100:.2f}%")
    
    print(f"\nFor {int(confidence_level*100)}% confidence level and ±{margin_of_error*100}% margin of error:")
    print(f"  Required sample (infinite population): {int(required_infinite):,}")
    print(f"  Required sample (finite population):   {int(required_finite):,}")
    
    if survey_n >= required_finite:
        print(f"\n✓ Sample size is ADEQUATE")
        actual_moe = calculate_margin_of_error(0.5, survey_n, confidence_level)
        print(f"  Actual margin of error: ±{actual_moe*100:.2f}%")
    else:
        print(f"\n⚠️  Sample size is INADEQUATE")
        shortfall = required_finite - survey_n
        print(f"  Shortfall: {int(shortfall):,} responses")
        actual_moe = calculate_margin_of_error(0.5, survey_n, confidence_level)
        print(f"  Actual margin of error: ±{actual_moe*100:.2f}%")

def coverage_analysis(survey_df, demographic_cols):
    """
    Check which demographic groups are covered and their sample sizes
    """
    print(f"\n{'='*80}")
    print(f"DEMOGRAPHIC COVERAGE ANALYSIS")
    print(f"{'='*80}")
    
    for col in demographic_cols:
        if col not in survey_df.columns:
            print(f"\n⚠️  Column '{col}' not found in survey")
            continue
            
        print(f"\n{col.upper().replace('_', ' ')}:")
        print(f"{'─'*60}")
        
        counts = survey_df[col].value_counts(dropna=False)
        props = survey_df[col].value_counts(normalize=True, dropna=False) * 100
        
        result = pd.DataFrame({
            'Count': counts,
            'Percent': props.round(1),
            'Sample_Adequate': counts >= 30  # Rule of thumb: n≥30 per group
        })
        
        print(result)
        
        missing = survey_df[col].isna().sum()
        if missing > 0:
            print(f"\n⚠️  Missing values: {missing} ({missing/len(survey_df)*100:.1f}%)")
        
        inadequate = (counts < 30).sum()
        if inadequate > 0:
            print(f"⚠️  {inadequate} categories have n < 30 (may have unreliable estimates)")

def calculate_weights(observed_props, target_props):
    """
    Calculate post-stratification weights to match target population
    
    Returns: DataFrame with weights for each category
    """
    weights = {}
    for cat in observed_props.index:
        if cat in target_props.index:
            weights[cat] = target_props[cat] / observed_props[cat]
        else:
            weights[cat] = 1.0
    
    return pd.Series(weights)

# ============================================================================
# 4. RUN REPRESENTATIVENESS TESTS
# ============================================================================

print("\n\n")
print("█" * 80)
print("SURVEY REPRESENTATIVENESS ANALYSIS")
print("█" * 80)

# Sample size adequacy
sample_size_analysis(len(survey), LSE_TOTAL_STUDENTS)

# Coverage analysis
coverage_cols = ['gender', 'year_of_study', 'department', 'lse_accommodation']
coverage_analysis(survey, coverage_cols)

# ============================================================================
# 5. CHI-SQUARE TESTS FOR EACH DEMOGRAPHIC
# ============================================================================

results_summary = []

# -------------------------
# GENDER
# -------------------------
if 'gender' in survey.columns:
    survey['gender_clean'] = clean_gender(survey['gender'])
    gender_counts = survey['gender_clean'].value_counts()
    
    # Only test Men/Women (exclude other categories for population comparison)
    gender_counts_binary = gender_counts[gender_counts.index.isin(['Men', 'Women'])]
    
    if len(gender_counts_binary) >= 2:
        chi2, p, v, comp = chi_square_goodness_of_fit(
            gender_counts_binary,
            LSE_DEMOGRAPHICS['gender'],
            'Gender'
        )
        results_summary.append({
            'Dimension': 'Gender',
            'Chi2': chi2,
            'P-value': p,
            'Cramers_V': v,
            'Representative': 'Yes' if p >= 0.05 else 'No'
        })

# -------------------------
# YEAR OF STUDY
# -------------------------
if 'year_of_study' in survey.columns:
    year_counts = survey['year_of_study'].value_counts()
    
    # You may need to map/aggregate categories to match LSE's reporting
    # For now, testing as-is
    chi2, p, v, comp = chi_square_goodness_of_fit(
        year_counts,
        LSE_DEMOGRAPHICS['year_of_study'],
        'Year of Study'
    )
    results_summary.append({
        'Dimension': 'Year of Study',
        'Chi2': chi2,
        'P-value': p,
        'Cramers_V': v,
        'Representative': 'Yes' if p >= 0.05 else 'No'
    })

# -------------------------
# DEPARTMENT
# -------------------------
if 'department' in survey.columns:
    dept_counts = survey['department'].value_counts()
    
    # Aggregate small departments into "Other" to match LSE demographics
    top_depts = list(LSE_DEMOGRAPHICS['department'].keys())
    top_depts.remove('Other')
    
    dept_counts_grouped = dept_counts.copy()
    other_count = dept_counts_grouped[~dept_counts_grouped.index.isin(top_depts)].sum()
    dept_counts_grouped = dept_counts_grouped[dept_counts_grouped.index.isin(top_depts)]
    dept_counts_grouped['Other'] = other_count
    
    chi2, p, v, comp = chi_square_goodness_of_fit(
        dept_counts_grouped,
        LSE_DEMOGRAPHICS['department'],
        'Department'
    )
    results_summary.append({
        'Dimension': 'Department',
        'Chi2': chi2,
        'P-value': p,
        'Cramers_V': v,
        'Representative': 'Yes' if p >= 0.05 else 'No'
    })

# ============================================================================
# 6. SUMMARY REPORT
# ============================================================================

print("\n\n")
print("█" * 80)
print("OVERALL REPRESENTATIVENESS SUMMARY")
print("█" * 80)

summary_df = pd.DataFrame(results_summary)
print("\n", summary_df.to_string(index=False))

representative_count = (summary_df['Representative'] == 'Yes').sum()
total_tests = len(summary_df)

print(f"\n{'='*80}")
print(f"VERDICT: {representative_count}/{total_tests} demographic dimensions are representative")
print(f"{'='*80}")

if representative_count == total_tests:
    print("\n✓ Sample appears REPRESENTATIVE of LSE population across all tested dimensions")
    print("  → Findings can be generalized to LSE with appropriate caveats")
    print("  → No weighting required")
elif representative_count >= total_tests * 0.5:
    print("\n⚠️  Sample is PARTIALLY REPRESENTATIVE")
    print("  → Some demographic imbalances detected")
    print("  → Consider post-stratification weights (see below)")
    print("  → Acknowledge limitations in reporting")
else:
    print("\n❌ Sample is NOT REPRESENTATIVE")
    print("  → Significant demographic imbalances detected")
    print("  → Post-stratification weights RECOMMENDED")
    print("  → Clearly state limitations: findings may not generalize to all LSE students")

# ============================================================================
# 7. WEIGHTING RECOMMENDATIONS
# ============================================================================

print("\n\n")
print("█" * 80)
print("POST-STRATIFICATION WEIGHTING RECOMMENDATIONS")
print("█" * 80)

print("\nIf you choose to weight your data to match LSE demographics:")

if 'gender' in survey.columns:
    print("\nGENDER WEIGHTS:")
    gender_props = survey['gender_clean'].value_counts(normalize=True)
    gender_props_binary = gender_props[gender_props.index.isin(['Men', 'Women'])]
    target_props = pd.Series(LSE_DEMOGRAPHICS['gender'])
    
    gender_weights = calculate_weights(gender_props_binary, target_props)
    print(gender_weights)
    
    print("\nTo apply:")
    print("  survey['gender_weight'] = survey['gender_clean'].map(gender_weights)")

# ============================================================================
# 8. VISUALIZATIONS
# ============================================================================

print("\n\nGenerating comparison visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Survey Sample vs LSE Population Demographics', fontsize=16, fontweight='bold')

# Gender comparison
if 'gender' in survey.columns:
    ax = axes[0, 0]
    
    sample_props = survey['gender_clean'].value_counts(normalize=True) * 100
    sample_props = sample_props[sample_props.index.isin(['Men', 'Women'])]
    
    lse_props = pd.Series({k: v*100 for k, v in LSE_DEMOGRAPHICS['gender'].items()})
    
    comparison = pd.DataFrame({
        'Survey Sample': sample_props,
        'LSE Population': lse_props
    })
    
    comparison.plot(kind='bar', ax=ax, color=['#2E86AB', '#F18F01'])
    ax.set_title('Gender Distribution', fontweight='bold')
    ax.set_ylabel('Percentage (%)')
    ax.set_xlabel('')
    ax.legend(title='')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

# Year of study comparison
if 'year_of_study' in survey.columns:
    ax = axes[0, 1]
    
    sample_props = survey['year_of_study'].value_counts(normalize=True) * 100
    lse_props = pd.Series({k: v*100 for k, v in LSE_DEMOGRAPHICS['year_of_study'].items()})
    
    # Match categories
    common = list(set(sample_props.index) & set(lse_props.index))
    comparison = pd.DataFrame({
        'Survey Sample': sample_props[common],
        'LSE Population': lse_props[common]
    })
    
    comparison.plot(kind='bar', ax=ax, color=['#2E86AB', '#F18F01'])
    ax.set_title('Year of Study Distribution', fontweight='bold')
    ax.set_ylabel('Percentage (%)')
    ax.set_xlabel('')
    ax.legend(title='')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

if 'department' in survey.columns:
    ax = axes[1, 0]
    
    sample_counts = survey['department'].value_counts()
    top_depts = list(LSE_DEMOGRAPHICS['department'].keys())
    top_depts.remove('Other')
    
    sample_grouped = sample_counts.copy()
    other_count = sample_grouped[~sample_grouped.index.isin(top_depts)].sum()
    sample_grouped = sample_grouped[sample_grouped.index.isin(top_depts)]
    sample_grouped['Other'] = other_count
    
    sample_props = sample_grouped / sample_grouped.sum() * 100
    lse_props = pd.Series({k: v*100 for k, v in LSE_DEMOGRAPHICS['department'].items()})
    
    comparison = pd.DataFrame({
        'Survey Sample': sample_props,
        'LSE Population': lse_props
    })
    
    comparison.plot(kind='bar', ax=ax, color=['#2E86AB', '#F18F01'])
    ax.set_title('Department Distribution', fontweight='bold')
    ax.set_ylabel('Percentage (%)')
    ax.set_xlabel('')
    ax.legend(title='')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

ax = axes[1, 1]
required = (1.96**2 * 0.5 * 0.5) / (0.05**2) / (1 + ((1.96**2 * 0.5 * 0.5) / (0.05**2) - 1) / LSE_TOTAL_STUDENTS)

categories = ['Required\nSample', 'Actual\nSample']
values = [required, len(survey)]
colors = ['#F18F01', '#2E86AB']

ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
ax.axhline(y=required, color='red', linestyle='--', linewidth=2, label=f'Minimum Required (n={int(required)})')
ax.set_title('Sample Size Adequacy', fontweight='bold')
ax.set_ylabel('Number of Responses')
ax.legend()

for i, v in enumerate(values):
    ax.text(i, v + max(values)*0.02, f'n = {int(v)}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations/representativeness_analysis.png', dpi=220, bbox_inches='tight')
plt.savefig('visualizations/representativeness_analysis.jpeg', dpi=220, bbox_inches='tight')
plt.show()

print("✓ Saved visualizations to: visualizations/representativeness_analysis.png")

# ============================================================================
# 9. EXPORT RESULTS
# ============================================================================

print("\n\nExporting results...")

# Save summary to CSV
summary_df.to_csv('representativeness_results.csv', index=False)
print("✓ Saved summary to: representativeness_results.csv")

# ============================================================================
# 10. RECOMMENDATIONS FOR REPORTING
# ============================================================================

print("\n\n")
print("█" * 80)
print("RECOMMENDATIONS FOR REPORTING")
print("█" * 80)

print("""
1. SAMPLE SIZE & RESPONSE RATE
   Include in your article:
   - Total LSE student population
   - Number of survey responses
   - Response rate (%)
   - Margin of error

2. DEMOGRAPHIC COMPARISON
   Report whether sample matches LSE population on:
   - Gender
   - Year of study  
   - Department
   Include chi-square test results if relevant

3. LIMITATIONS STATEMENT
   Example text:
   "This survey received [N] responses from LSE students, representing a [X]% 
   response rate. The sample was [representative/not fully representative] of 
   the LSE population in terms of [dimensions]. Results should be interpreted 
   with this in mind, particularly for [specific groups]."

4. IF USING WEIGHTS
   State clearly:
   "To ensure findings are representative of the broader LSE population, 
   we applied post-stratification weights based on gender and year of study."

5. CONFIDENCE INTERVALS
   For key statistics (e.g., "X% of students have..."), report 95% CI:
   proportion ± margin_of_error
""")

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)