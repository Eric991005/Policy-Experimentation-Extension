#!/usr/bin/env python3
import json
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from linearmodels.panel import PanelOLS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE = Path('/root/autodl-tmp/policy_data_collection/project_extension')
DATA_FINAL = BASE / 'data_final'
TABLES = BASE / 'outputs' / 'tables'
FIGS = BASE / 'outputs' / 'figures'
MEMO = BASE / 'outputs' / 'memo' / 'extension_research_memo.md'
LOG = BASE / 'logs' / 'pipeline.log'
MAN = BASE / 'manifests'

for d in [DATA_FINAL, TABLES, FIGS, MAN]:
    d.mkdir(parents=True, exist_ok=True)


def log(msg):
    line = f"{datetime.now().isoformat(timespec='seconds')} [ENHANCE] {msg}"
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def nbs_query_indicator(indicator_code: str):
    params = {
        'm': 'QueryData',
        'dbcode': 'fsnd',
        'rowcode': 'reg',
        'colcode': 'sj',
        'wds': '[]',
        'dfwds': json.dumps([{"wdcode": "zb", "valuecode": indicator_code}], ensure_ascii=False),
        'k': '1'
    }
    r = requests.get('https://data.stats.gov.cn/easyquery.htm', params=params, timeout=40)
    r.raise_for_status()
    j = r.json()
    if j.get('returncode') != 200:
        return None, None, None

    rd = j['returndata']
    zb_nodes = [x for x in rd['wdnodes'] if x['wdcode'] == 'zb'][0]['nodes']
    reg_nodes = [x for x in rd['wdnodes'] if x['wdcode'] == 'reg'][0]['nodes']
    reg_map = {n['code']: n['cname'] for n in reg_nodes}

    ind_name = zb_nodes[0]['cname'] if zb_nodes else indicator_code
    ind_unit = zb_nodes[0].get('unit', '') if zb_nodes else ''

    rows = []
    for dn in rd.get('datanodes', []):
        wds = {w['wdcode']: w['valuecode'] for w in dn.get('wds', [])}
        reg = wds.get('reg')
        year = wds.get('sj')
        val_raw = dn.get('data', {}).get('strdata', '')
        if val_raw is None:
            val_raw = ''
        val_raw = str(val_raw).replace(',', '').strip()
        try:
            val = float(val_raw) if val_raw not in ['', '—', '-', '...'] else np.nan
        except Exception:
            val = np.nan
        rows.append({
            'province_code': reg,
            'province_name': reg_map.get(reg, ''),
            'year': int(year) if str(year).isdigit() else np.nan,
            'indicator_code': indicator_code,
            'indicator_name': ind_name,
            'indicator_unit': ind_unit,
            'value': val,
        })

    return pd.DataFrame(rows), ind_name, ind_unit


def normalize_province_name(s: str) -> str:
    s = str(s or '').strip()
    repl = ['壮族自治区', '回族自治区', '维吾尔自治区', '自治区', '省', '市', '特别行政区']
    for r in repl:
        s = s.replace(r, '')
    return s


def save_regression_result(res, name):
    out = pd.DataFrame({
        'term': res.params.index,
        'coef': res.params.values,
        'std_err': res.std_errors.values,
        't': res.tstats.values,
        'p_value': res.pvalues.values,
    })
    out['model'] = name
    out['nobs'] = float(res.nobs)
    out['rsquared'] = float(getattr(res, 'rsquared', np.nan))
    out.to_csv(TABLES / f'regression_results_{name}.csv', index=False)
    out.to_latex(TABLES / f'regression_results_{name}.tex', index=False, float_format='%.4f')
    with (TABLES / f'regression_summary_{name}.txt').open('w', encoding='utf-8') as f:
        f.write(str(res.summary))
    return out


def main():
    log('Start enhanced macro integration')

    indicator_map = {
        'GDP': 'A020101',
        'GDP_per_capita': 'A02010G',
        'SecondIndustry': 'A020103',
        'ThirdIndustry': 'A020104',
        'Population': 'A030101',
        'FiscalRevenue': 'A080101',
        'FiscalExpenditure': 'A080201',
        'FAI_growth': 'A050101',
    }

    long_rows = []
    meta = []
    for var, code in indicator_map.items():
        try:
            df, name, unit = nbs_query_indicator(code)
            if df is None or df.empty:
                log(f'NBS indicator {var}({code}) empty')
                continue
            df['var_name'] = var
            long_rows.append(df)
            meta.append({'var_name': var, 'indicator_code': code, 'indicator_name': name, 'unit': unit, 'rows': len(df)})
            log(f'NBS indicator {var}({code}) fetched rows={len(df)}')
        except Exception as e:
            log(f'NBS indicator {var}({code}) failed: {e}')

    if not long_rows:
        raise RuntimeError('No NBS indicators fetched')

    long_df = pd.concat(long_rows, ignore_index=True)
    long_df = long_df.dropna(subset=['year'])
    long_df['year'] = long_df['year'].astype(int)

    macro = (
        long_df
        .pivot_table(index=['province_code', 'province_name', 'year'], columns='var_name', values='value', aggfunc='mean')
        .reset_index()
    )
    macro.columns.name = None

    if 'GDP' in macro.columns and 'SecondIndustry' in macro.columns:
        macro['ShareSecondary'] = macro['SecondIndustry'] / macro['GDP']
    if 'GDP' in macro.columns and 'ThirdIndustry' in macro.columns:
        macro['ShareTertiary'] = macro['ThirdIndustry'] / macro['GDP']

    macro['province_norm'] = macro['province_name'].map(normalize_province_name)
    macro = macro.sort_values(['province_norm', 'year'])
    if 'GDP' in macro.columns:
        macro['GDP_growth'] = macro.groupby('province_norm')['GDP'].pct_change()
    if 'GDP_per_capita' in macro.columns:
        macro['GDP_pc_growth'] = macro.groupby('province_norm')['GDP_per_capita'].pct_change()

    macro_csv = DATA_FINAL / 'macro_province_year_from_nbs.csv'
    macro.to_csv(macro_csv, index=False)
    macro.to_parquet(DATA_FINAL / 'macro_province_year_from_nbs.parquet', index=False)
    (MAN / 'nbs_macro_indicator_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    log(f'Macro saved: {macro_csv}, rows={len(macro)}')

    # Merge to city-year panel
    city_year = pd.read_csv(DATA_FINAL / 'city_year_panel.csv')
    city_year['province_norm'] = city_year['province_name'].map(normalize_province_name)
    enh = city_year.merge(
        macro,
        on=['province_norm', 'year'],
        how='left',
        suffixes=('', '_macro')
    )

    # overwrite placeholder macro columns in city_year_panel with newly fetched macro values
    macro_cols = ['GDP','GDP_per_capita','Population','FiscalRevenue','FiscalExpenditure',
                  'SecondIndustry','ThirdIndustry','ShareSecondary','ShareTertiary',
                  'GDP_growth','GDP_pc_growth','FAI_growth','province_code','province_name']
    for c in macro_cols:
        cm = f'{c}_macro'
        if cm in enh.columns:
            enh[c] = enh[cm]

    # outcome t+1
    enh = enh.sort_values(['province_norm', 'year'])
    if 'GDP_pc_growth' in enh.columns:
        enh['Outcome_t1'] = enh.groupby('province_norm')['GDP_pc_growth'].shift(-1)
    else:
        enh['Outcome_t1'] = np.nan

    enh.to_csv(DATA_FINAL / 'city_year_panel_enhanced_macro.csv', index=False)
    enh.to_parquet(DATA_FINAL / 'city_year_panel_enhanced_macro.parquet', index=False)

    # Enhanced Model B
    keep_cols = ['city_name', 'year', 'Outcome_t1', 'PolicyResponseIntensity', 'alignment_score', 'PostCount',
                 'GDP_growth', 'GDP_pc_growth', 'FiscalRevenue', 'FiscalExpenditure', 'Population',
                 'ShareSecondary', 'ShareTertiary']
    for c in keep_cols:
        if c not in enh.columns:
            enh[c] = np.nan

    reg0 = enh[keep_cols].copy()
    reg0['time_id'] = pd.to_datetime(reg0['year'].astype(int).astype(str) + '-01-01')
    reg0['Capacity'] = np.log1p(reg0['PostCount'])

    x_cols = ['PolicyResponseIntensity', 'alignment_score', 'Capacity',
              'GDP_growth', 'FiscalRevenue', 'FiscalExpenditure', 'Population', 'ShareSecondary']

    reg0 = reg0.dropna(subset=['Outcome_t1', 'city_name', 'time_id'])
    reg0 = reg0.dropna(subset=[c for c in x_cols if c in reg0.columns])
    log(f'Enhanced regression candidate rows={len(reg0)}')
    if len(reg0) == 0:
        raise RuntimeError('No non-missing rows for enhanced regression after merge/dropna')

    # drop constant columns to avoid rank errors
    x_cols_eff = [c for c in x_cols if reg0[c].nunique(dropna=True) > 1]
    dropped = sorted(set(x_cols) - set(x_cols_eff))
    if dropped:
        log(f'Dropped constant/degenerate controls: {dropped}')
    if len(x_cols_eff) == 0:
        # minimal fallback
        x_cols_eff = [c for c in ['PolicyResponseIntensity', 'alignment_score'] if c in reg0.columns]
        log(f'All controls dropped; fallback regressors={x_cols_eff}')

    reg = reg0.set_index(['city_name', 'time_id']).sort_index()
    y = reg['Outcome_t1']
    X = reg[x_cols_eff]

    # try PanelOLS first; fallback to OLS with FE dummies if linearmodels fails
    res = None
    out = None
    try:
        res = PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True, check_rank=False).fit(
            cov_type='clustered', cluster_entity=True
        )
        out = save_regression_result(res, 'model_B_enhanced')
        model_used = 'PanelOLS'
    except Exception as e:
        log(f'PanelOLS failed, fallback to OLS_FE: {e}')
        import statsmodels.formula.api as smf
        regf = reg0.copy()
        regf['year_cat'] = regf['year'].astype(int).astype(str)
        rhs = ' + '.join(x_cols_eff) + ' + C(city_name) + C(year_cat)'
        formula = f'Outcome_t1 ~ {rhs}'
        ols = smf.ols(formula, data=regf).fit(cov_type='cluster', cov_kwds={'groups': regf['city_name']})
        rows = []
        for term in x_cols_eff:
            rows.append({
                'term': term,
                'coef': float(ols.params.get(term, np.nan)),
                'std_err': float(ols.bse.get(term, np.nan)),
                't': float(ols.tvalues.get(term, np.nan)),
                'p_value': float(ols.pvalues.get(term, np.nan)),
                'model': 'model_B_enhanced_ols_fe',
                'nobs': float(ols.nobs),
                'rsquared': float(getattr(ols, 'rsquared', np.nan)),
            })
        out = pd.DataFrame(rows)
        out.to_csv(TABLES / 'regression_results_model_B_enhanced.csv', index=False)
        out.to_latex(TABLES / 'regression_results_model_B_enhanced.tex', index=False, float_format='%.4f')
        with (TABLES / 'regression_summary_model_B_enhanced.txt').open('w', encoding='utf-8') as f:
            f.write(str(ols.summary()))
        model_used = 'OLS_FE'


    # coef figure
    out['ci_low'] = out['coef'] - 1.96 * out['std_err']
    out['ci_high'] = out['coef'] + 1.96 * out['std_err']
    plt.figure(figsize=(9, 5))
    sns.pointplot(data=out, y='term', x='coef', join=False)
    plt.axvline(0, color='black', lw=1)
    plt.title('Model B Enhanced Coefficients (Outcome_t1 = GDP per capita growth lead)')
    plt.tight_layout()
    plt.savefig(FIGS / '06_model_B_enhanced_coefficients.png', dpi=150)
    plt.close()

    # append memo
    addon = []
    addon.append('\n## 8. 增强版更新（省级宏观并表）')
    addon.append(f'- 更新时间: {datetime.now().isoformat(timespec="seconds")}')
    addon.append('- 新增数据: data.stats.gov.cn(fsnd) 省级年度宏观指标（GDP、人均GDP、人口、财政收支、二三产结构、固定资产投资增速）')
    addon.append('- 新增文件:')
    addon.append('  - data_final/macro_province_year_from_nbs.csv')
    addon.append('  - data_final/city_year_panel_enhanced_macro.csv')
    addon.append('  - outputs/tables/regression_results_model_B_enhanced.csv')
    addon.append('  - outputs/figures/06_model_B_enhanced_coefficients.png')
    nobs_val = int(out['nobs'].iloc[0]) if len(out) else 0
    addon.append(f'- Enhanced Model B样本量: {nobs_val}')
    addon.append(f'- Enhanced Model B估计器: {model_used}')
    addon.append('- 说明: 当前宏观变量为“省级并表到城市”，可用于增强稳健性，但非纯地级市硬结果变量。')

    if MEMO.exists():
        with MEMO.open('a', encoding='utf-8') as f:
            f.write('\n'.join(addon) + '\n')
    else:
        MEMO.write_text('\n'.join(addon) + '\n', encoding='utf-8')

    summary = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'macro_rows': int(len(macro)),
        'enhanced_city_year_rows': int(len(enh)),
        'model_B_enhanced_estimator': model_used,
        'model_B_enhanced_nobs': float(out['nobs'].iloc[0]) if len(out) else 0.0,
        'model_B_enhanced_rsquared': float(out['rsquared'].iloc[0]) if len(out) else np.nan,
        'model_B_enhanced_terms': out.to_dict(orient='records')
    }
    (MAN / 'enhanced_run_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    log('Enhanced macro + Model B done')


if __name__ == '__main__':
    main()
