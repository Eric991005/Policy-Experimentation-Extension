#!/usr/bin/env python3
import json
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path('/root/autodl-tmp/policy_data_collection/project_extension')
DATA_FINAL = BASE / 'data_final'
TABLES = BASE / 'outputs' / 'tables'
FIGS = BASE / 'outputs' / 'figures'
LOG = BASE / 'logs' / 'pipeline.log'
MEMO = BASE / 'outputs' / 'memo' / 'extension_research_memo.md'
MAN = BASE / 'manifests'

for d in [TABLES, FIGS, MAN]:
    d.mkdir(parents=True, exist_ok=True)


def log(msg):
    line = f"{datetime.now().isoformat(timespec='seconds')} [EVENT] {msg}"
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def parse_date_from_source_url(url: str):
    u = str(url or '')
    # /2021-08/27/
    m = re.search(r'/(19\d{2}|20\d{2})-(\d{2})/(\d{2})/', u)
    if m:
        try:
            return pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=int(m.group(3)))
        except Exception:
            pass
    # /2021/08/27/
    m = re.search(r'/(19\d{2}|20\d{2})/(\d{2})/(\d{2})/', u)
    if m:
        try:
            return pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=int(m.group(3)))
        except Exception:
            pass
    # /2021-08/
    m = re.search(r'/(19\d{2}|20\d{2})-(\d{2})/', u)
    if m:
        try:
            return pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=1)
        except Exception:
            pass
    return pd.NaT


def build_central_timeline(city_month: pd.DataFrame, central: pd.DataFrame) -> pd.DataFrame:
    c = central.copy()
    c['date_raw'] = pd.to_datetime(c['date'], errors='coerce')
    c['date_url'] = c['source_url'].map(parse_date_from_source_url)
    c['date_final'] = c['date_url']
    c.loc[c['date_final'].isna(), 'date_final'] = c.loc[c['date_final'].isna(), 'date_raw']
    c = c.dropna(subset=['date_final']).copy()

    score_cols = ['RiskScore','StabilityScore','ScalabilityScore','EvalPressureScore','StrategicPriorityScore','ConditionalityScore']
    for sc in score_cols:
        c[sc] = pd.to_numeric(c[sc], errors='coerce').fillna(0)
    c['ObjectiveIndex'] = c[['RiskScore','StabilityScore','ScalabilityScore','EvalPressureScore','StrategicPriorityScore']].mean(axis=1) / 5.0

    c['month_key'] = c['date_final'].dt.to_period('M').astype(str)

    m = c.groupby('month_key', as_index=False).agg(
        docs_count=('title','count'),
        ObjectiveIndex=('ObjectiveIndex','mean'),
        RiskIndex=('RiskScore', lambda s: s.mean()/5.0),
        StabilityIndex=('StabilityScore', lambda s: s.mean()/5.0),
        ScalabilityIndex=('ScalabilityScore', lambda s: s.mean()/5.0),
        EvalIndex=('EvalPressureScore', lambda s: s.mean()/5.0),
        StrategicIndex=('StrategicPriorityScore', lambda s: s.mean()/5.0),
        ConditionalityIndex=('ConditionalityScore', lambda s: s.mean()/5.0),
    )

    # expand full month index
    month_min = pd.Period(city_month['month_key'].min(), freq='M')
    month_max = pd.Period(city_month['month_key'].max(), freq='M')
    full = pd.DataFrame({'month_key': [str(p) for p in pd.period_range(month_min, month_max, freq='M')]})
    m = full.merge(m, on='month_key', how='left').sort_values('month_key')
    m['docs_count'] = m['docs_count'].fillna(0)

    for col in ['ObjectiveIndex','RiskIndex','StabilityIndex','ScalabilityIndex','EvalIndex','StrategicIndex','ConditionalityIndex']:
        m[col] = m[col].fillna(0)

    # event definition
    thresh = float(m['ObjectiveIndex'].quantile(0.9)) if len(m) > 0 else 0.0
    m['EventAny'] = (m['docs_count'] > 0).astype(int)
    m['EventHigh'] = ((m['docs_count'] > 0) & (m['ObjectiveIndex'] >= thresh)).astype(int)
    if m['EventHigh'].sum() == 0:
        # fallback if threshold too strict
        m['EventHigh'] = m['EventAny']

    m.to_csv(DATA_FINAL / 'central_event_timeline_monthly.csv', index=False)
    return m


def run_event_window_model(city_month: pd.DataFrame, timeline: pd.DataFrame):
    df = city_month.copy()
    df = df.merge(timeline, on='month_key', how='left')
    df['Capacity'] = np.log1p(df.groupby('city_name')['PostCount'].transform('mean'))

    # build national event pulse on month index
    t = timeline[['month_key','EventHigh']].copy()
    t['period'] = pd.PeriodIndex(t['month_key'], freq='M')
    t = t.sort_values('period').reset_index(drop=True)

    window = list(range(-6, 7))
    if -1 in window:
        window.remove(-1)  # baseline

    for k in window:
        if k < 0:
            s = t['EventHigh'].shift(-k)  # lead
        else:
            s = t['EventHigh'].shift(k)   # lag
        t[f'W{k}'] = s.fillna(0)

    use_cols = ['month_key'] + [f'W{k}' for k in window]
    df = df.merge(t[use_cols], on='month_key', how='left')

    xcols = []
    for k in window:
        v = f'W{k}_xCap'
        df[v] = df[f'W{k}'] * df['Capacity']
        if df[v].nunique(dropna=True) > 1:
            xcols.append(v)
    if len(xcols) == 0:
        raise RuntimeError('No valid regressors in event-window model after variation check')

    df['time_id'] = pd.to_datetime(df['year'].astype(int).astype(str) + '-' + df['month'].astype(int).astype(str).str.zfill(2) + '-01')
    reg = df.dropna(subset=['PolicyResponseIntensity','city_name','time_id'])
    reg = reg.set_index(['city_name','time_id']).sort_index()

    y = reg['PolicyResponseIntensity']
    X = reg[xcols]

    res = PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True, check_rank=False).fit(
        cov_type='clustered', cluster_entity=True
    )

    out = pd.DataFrame({
        'term': res.params.index,
        'coef': res.params.values,
        'std_err': res.std_errors.values,
        't': res.tstats.values,
        'p_value': res.pvalues.values,
        'model': 'model_D_event_window',
        'nobs': float(res.nobs),
        'rsquared': float(getattr(res, 'rsquared', np.nan))
    })
    out['event_k'] = out['term'].str.extract(r'W(-?\d+)_xCap').astype(float)

    out.to_csv(TABLES / 'regression_results_model_D_event_window.csv', index=False)
    out.to_latex(TABLES / 'regression_results_model_D_event_window.tex', index=False, float_format='%.4f')
    with (TABLES / 'regression_summary_model_D_event_window.txt').open('w', encoding='utf-8') as f:
        f.write(str(res.summary))

    # figure
    fig_df = out.dropna(subset=['event_k']).sort_values('event_k')
    fig_df['ci_low'] = fig_df['coef'] - 1.96 * fig_df['std_err']
    fig_df['ci_high'] = fig_df['coef'] + 1.96 * fig_df['std_err']
    plt.figure(figsize=(9,5))
    plt.errorbar(fig_df['event_k'], fig_df['coef'], yerr=1.96*fig_df['std_err'], fmt='o-')
    plt.axhline(0, color='black', lw=1)
    plt.axvline(-1, color='gray', ls='--', lw=1)
    plt.title('Dynamic Event Window Coefficients (x Capacity)')
    plt.xlabel('Relative month k (baseline = -1)')
    plt.ylabel('Coefficient')
    plt.tight_layout()
    plt.savefig(FIGS / '07_event_window_coefficients.png', dpi=150)
    plt.close()

    return out


def run_lag_model(city_month: pd.DataFrame, timeline: pd.DataFrame):
    df = city_month.copy()
    t = timeline[['month_key','ObjectiveIndex']].copy()
    t['period'] = pd.PeriodIndex(t['month_key'], freq='M')
    t = t.sort_values('period').reset_index(drop=True)

    for l in range(0, 7):
        t[f'ObjLag{l}'] = t['ObjectiveIndex'].shift(l).fillna(0)

    df = df.merge(t[['month_key'] + [f'ObjLag{l}' for l in range(0, 7)]], on='month_key', how='left')
    df['Capacity'] = np.log1p(df.groupby('city_name')['PostCount'].transform('mean'))

    xcols = []
    for l in range(0, 7):
        v = f'ObjLag{l}_xCap'
        df[v] = df[f'ObjLag{l}'] * df['Capacity']
        if df[v].nunique(dropna=True) > 1:
            xcols.append(v)
    if len(xcols) == 0:
        raise RuntimeError('No valid regressors in lag model after variation check')

    df['time_id'] = pd.to_datetime(df['year'].astype(int).astype(str) + '-' + df['month'].astype(int).astype(str).str.zfill(2) + '-01')
    reg = df.dropna(subset=['PolicyResponseIntensity','city_name','time_id'])
    reg = reg.set_index(['city_name','time_id']).sort_index()

    y = reg['PolicyResponseIntensity']
    X = reg[xcols]

    res = PanelOLS(y, X, entity_effects=True, time_effects=True, drop_absorbed=True, check_rank=False).fit(
        cov_type='clustered', cluster_entity=True
    )

    out = pd.DataFrame({
        'term': res.params.index,
        'coef': res.params.values,
        'std_err': res.std_errors.values,
        't': res.tstats.values,
        'p_value': res.pvalues.values,
        'model': 'model_E_lag',
        'nobs': float(res.nobs),
        'rsquared': float(getattr(res, 'rsquared', np.nan))
    })
    out['lag_l'] = out['term'].str.extract(r'ObjLag(\d+)_xCap').astype(float)

    out.to_csv(TABLES / 'regression_results_model_E_lag.csv', index=False)
    out.to_latex(TABLES / 'regression_results_model_E_lag.tex', index=False, float_format='%.4f')
    with (TABLES / 'regression_summary_model_E_lag.txt').open('w', encoding='utf-8') as f:
        f.write(str(res.summary))

    fig_df = out.dropna(subset=['lag_l']).sort_values('lag_l')
    plt.figure(figsize=(9,5))
    plt.errorbar(fig_df['lag_l'], fig_df['coef'], yerr=1.96*fig_df['std_err'], fmt='o-')
    plt.axhline(0, color='black', lw=1)
    plt.title('Distributed Lag Coefficients of ObjectiveIndex (x Capacity)')
    plt.xlabel('Lag l (months)')
    plt.ylabel('Coefficient')
    plt.tight_layout()
    plt.savefig(FIGS / '08_lag_effects_coefficients.png', dpi=150)
    plt.close()

    return out


def append_memo(event_out: pd.DataFrame, lag_out: pd.DataFrame):
    lines = []
    lines.append('\n## 9. 动态事件窗/滞后模型（新增）')
    lines.append(f'- 更新时间: {datetime.now().isoformat(timespec="seconds")}')
    lines.append('- 已新增 central_event_timeline_monthly.csv 并并入 city_month_panel。')
    lines.append('- Model D（事件窗）: EventHigh 的 lead/lag 与 Capacity 交互，城市FE+月FE。')
    lines.append('- Model E（分布式滞后）: ObjectiveIndex 的 0-6 月滞后与 Capacity 交互，城市FE+月FE。')

    topD = event_out.sort_values('p_value').head(5)
    topE = lag_out.sort_values('p_value').head(5)
    lines.append('- Event-window 最显著系数（前5）：')
    for _, r in topD.iterrows():
        lines.append(f"  - {r['term']}: coef={r['coef']:.4f}, p={r['p_value']:.4f}")
    lines.append('- Lag-model 最显著系数（前5）：')
    for _, r in topE.iterrows():
        lines.append(f"  - {r['term']}: coef={r['coef']:.4f}, p={r['p_value']:.4f}")

    if MEMO.exists():
        with MEMO.open('a', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
    else:
        MEMO.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    log('start event timeline + dynamic models')
    city_month = pd.read_csv(DATA_FINAL / 'city_month_panel.csv')
    central = pd.read_csv(DATA_FINAL / 'central_policy_objectives.csv')

    timeline = build_central_timeline(city_month, central)
    log(f'timeline months={len(timeline)} event_any={int(timeline.EventAny.sum())} event_high={int(timeline.EventHigh.sum())}')

    event_out = run_event_window_model(city_month, timeline)
    log(f'model_D done terms={len(event_out)} nobs={float(event_out.nobs.iloc[0]) if len(event_out) else 0}')

    lag_out = run_lag_model(city_month, timeline)
    log(f'model_E done terms={len(lag_out)} nobs={float(lag_out.nobs.iloc[0]) if len(lag_out) else 0}')

    append_memo(event_out, lag_out)

    summary = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'timeline_rows': int(len(timeline)),
        'event_any_months': int(timeline['EventAny'].sum()),
        'event_high_months': int(timeline['EventHigh'].sum()),
        'model_D_terms': event_out.to_dict(orient='records'),
        'model_E_terms': lag_out.to_dict(orient='records'),
    }
    (MAN / 'event_lag_models_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    log('done event timeline + dynamic models')


if __name__ == '__main__':
    main()
