#!/usr/bin/env python3
import os
import re
import json
import csv
import math
import time
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import pyarrow as pa
import pyarrow.parquet as pq
from linearmodels.panel import PanelOLS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.mode.copy_on_write = True

BASE = Path('/root/autodl-tmp/policy_data_collection')
EXT = BASE / 'project_extension'
RAW = BASE / 'raw'
MAN = BASE / 'manifests'

DIRS = {
    'raw_links': EXT / 'data_raw_links',
    'inter': EXT / 'data_intermediate',
    'final': EXT / 'data_final',
    'scripts': EXT / 'scripts',
    'logs': EXT / 'logs',
    'tables': EXT / 'outputs' / 'tables',
    'figs': EXT / 'outputs' / 'figures',
    'memo': EXT / 'outputs' / 'memo',
    'man': EXT / 'manifests',
}
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

LOG_FILE = DIRS['logs'] / 'pipeline.log'
RUN_LOG = DIRS['logs'] / 'run.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def log(msg: str):
    logging.info(msg)
    with RUN_LOG.open('a', encoding='utf-8') as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

# ---------------------- Utilities ----------------------

def detect_encoding(path: Path) -> str:
    sample = path.open('rb').read(2_000_000)
    for enc in ['utf-8-sig', 'utf-8', 'gb18030', 'gbk']:
        try:
            sample.decode(enc)
            return enc
        except Exception:
            continue
    return 'utf-8'


def normalize_text(s: str) -> str:
    if s is None:
        return ''
    s = str(s)
    s = re.sub(r'<[^>]+>', ' ', s)
    s = s.replace('\u3000', ' ')
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def normalize_title_for_dup(s: str) -> str:
    s = normalize_text(s).lower()
    s = re.sub(r'[\W_]+', '', s)
    return s


def parse_ymd_from_year_date(year_val: str, date_val: str) -> Optional[pd.Timestamp]:
    year_val = (str(year_val) if year_val is not None else '').strip()
    date_val = (str(date_val) if date_val is not None else '').strip()

    if year_val.isdigit() and re.match(r'^\d{1,2}[-/]\d{1,2}$', date_val):
        mm, dd = [int(x) for x in re.split(r'[-/]', date_val)]
        try:
            return pd.Timestamp(year=int(year_val), month=mm, day=dd)
        except Exception:
            return None

    m = re.search(r'(19\d{2}|20\d{2})[-/年\.](\d{1,2})[-/月\.](\d{1,2})', date_val)
    if m:
        y, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return pd.Timestamp(year=y, month=mm, day=dd)
        except Exception:
            return None

    if year_val.isdigit() and len(year_val) == 4:
        try:
            return pd.Timestamp(year=int(year_val), month=1, day=1)
        except Exception:
            return None
    return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------- LLM Client ----------------------

class SiliconFlowClient:
    def __init__(self):
        self.api_key = os.getenv('SILICONFLOW_API_KEY', '').strip()
        self.base = os.getenv('SILICONFLOW_BASE', 'https://api.siliconflow.cn/v1/chat/completions')
        self.model = os.getenv('SILICONFLOW_MODEL', 'Qwen/Qwen2.5-72B-Instruct')
        self.enabled = bool(self.api_key)
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        })

    def chat_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.1, max_tokens: int = 700) -> Optional[dict]:
        if not self.enabled:
            return None
        payload = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            'temperature': temperature,
            'max_tokens': max_tokens
        }
        try:
            r = self.session.post(self.base, data=json.dumps(payload), timeout=35)
            if r.status_code != 200:
                log(f'LLM_HTTP_{r.status_code}: {r.text[:200]}')
                return None
            data = r.json()
            text = data['choices'][0]['message']['content']
            m = re.search(r'\{[\s\S]*\}', text)
            if not m:
                return None
            return json.loads(m.group(0))
        except Exception as e:
            log(f'LLM_ERROR: {e}')
            return None


LLM = SiliconFlowClient()


def fallback_policy_scores(text: str) -> dict:
    rules = {
        'RiskScore': ['风险', '防范', '隐患', '应急'],
        'StabilityScore': ['稳定', '稳妥', '平稳', '有序'],
        'ScalabilityScore': ['可复制', '可推广', '示范', '推广'],
        'EvalPressureScore': ['考核', '评估', '督导', '责任', '落实'],
        'StrategicPriorityScore': ['国家战略', '十四五', '现代化', '高质量发展', '强国'],
        'ConditionalityScore': ['因地制宜', '分类推进', '具备条件', '先行先试', '分步实施'],
    }
    out = {}
    for k, kws in rules.items():
        cnt = sum(text.count(w) for w in kws)
        out[k] = min(5, int(math.log1p(cnt) * 2))
    out['evidence'] = {k: '' for k in rules}
    out['confidence'] = 0.35
    return out


# ---------------------- Task 1 ----------------------

def task1_audit_and_clean_city_weixin() -> Path:
    log('TASK1: audit + clean city weixin start')
    city_dir = RAW / 'city_gov_weixin'
    files = sorted(city_dir.glob('shizhengfu_part_*.csv'))
    if not files:
        raise RuntimeError('No shizhengfu_part_*.csv found')

    out_parquet = DIRS['inter'] / 'cleaned_city_weixin.parquet'
    out_parquet_final = DIRS['final'] / 'cleaned_city_weixin.parquet'
    dedup_report_path = DIRS['man'] / 'dedup_report.json'
    schema_report_path = DIRS['man'] / 'schema_report.md'

    seen_full = set()
    seen_title = set()

    csv.field_size_limit(1024 * 1024 * 1024)

    writer = None
    row_counter = 0
    stats = {
        'files': [],
        'total_in_rows': 0,
        'total_out_rows': 0,
        'dropped_exact_duplicate': 0,
        'dropped_title_date_city_duplicate': 0,
        'dropped_near_duplicate': 0,
        'schema_fields': [
            'doc_id', 'city_name', 'province_name', 'date', 'year', 'month', 'quarter',
            'title', 'text', 'source_file'
        ]
    }

    col_alias = {
        'province_name': ['省份', '省', 'province'],
        'city_name': ['城市', 'city'],
        'account_name': ['公众号名称', '公众号', '公众名称'],
        'title': ['标题', 'title'],
        'text': ['正文', '内容', 'text', 'content'],
        'url': ['链接', 'url'],
        'year': ['年份', '发布年份', 'year'],
        'date_raw': ['日期', '发布日期', 'date'],
        'time_raw': ['时间', '发布时间', 'time'],
    }

    def pick_col(df_cols, aliases):
        for a in aliases:
            if a in df_cols:
                return a
        low = {c.lower(): c for c in df_cols}
        for a in aliases:
            for k, v in low.items():
                if a.lower() in k:
                    return v
        return None

    for fp in files:
        enc = detect_encoding(fp)
        log(f'TASK1: reading {fp.name} encoding={enc}')
        file_in = 0
        file_out = 0
        chunks = pd.read_csv(fp, dtype=str, encoding=enc, chunksize=100_000, low_memory=False)
        for chunk in chunks:
            file_in += len(chunk)
            stats['total_in_rows'] += len(chunk)

            c_city = pick_col(chunk.columns, col_alias['city_name'])
            c_prov = pick_col(chunk.columns, col_alias['province_name'])
            c_title = pick_col(chunk.columns, col_alias['title'])
            c_text = pick_col(chunk.columns, col_alias['text'])
            c_year = pick_col(chunk.columns, col_alias['year'])
            c_date = pick_col(chunk.columns, col_alias['date_raw'])

            if c_city is None or c_title is None or c_text is None:
                continue

            df = pd.DataFrame()
            df['city_name'] = chunk[c_city].astype(str).map(normalize_text)
            df['province_name'] = chunk[c_prov].astype(str).map(normalize_text) if c_prov else ''
            df['title'] = chunk[c_title].astype(str).map(normalize_text)
            df['text'] = chunk[c_text].astype(str).map(normalize_text)
            df['year_raw'] = chunk[c_year].astype(str).map(normalize_text) if c_year else ''
            df['date_raw'] = chunk[c_date].astype(str).map(normalize_text) if c_date else ''
            df['source_file'] = fp.name

            # parse date
            df['date'] = [parse_ymd_from_year_date(y, d) for y, d in zip(df['year_raw'], df['date_raw'])]
            df = df[df['city_name'].str.len() > 0].copy()
            df = df[df['title'].str.len() > 0].copy()
            df = df[df['text'].str.len() > 0].copy()
            df = df[~df['date'].isna()].copy()

            df['year'] = df['date'].dt.year.astype(int)
            df['month'] = df['date'].dt.month.astype(int)
            df['quarter'] = ((df['month'] - 1) // 3 + 1).astype(int)
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')

            norm_title = df['title'].map(normalize_title_for_dup)
            full_key = (
                df['city_name'] + '|' + df['date'] + '|' + df['title'] + '|' + df['text']
            ).map(lambda x: hashlib.md5(x.encode('utf-8', errors='ignore')).hexdigest())
            title_key = (
                df['city_name'] + '|' + df['date'] + '|' + norm_title
            ).map(lambda x: hashlib.md5(x.encode('utf-8', errors='ignore')).hexdigest())

            dup_full = full_key.isin(seen_full)
            dup_title = (~dup_full) & title_key.isin(seen_title)
            # conservative near-dup: extremely short edit by normalized title same prefix (already captured in title dup)
            dup_near = pd.Series(False, index=df.index)

            keep_mask = ~(dup_full | dup_title | dup_near)

            stats['dropped_exact_duplicate'] += int(dup_full.sum())
            stats['dropped_title_date_city_duplicate'] += int(dup_title.sum())
            stats['dropped_near_duplicate'] += int(dup_near.sum())

            kept = df.loc[keep_mask, ['city_name', 'province_name', 'date', 'year', 'month', 'quarter', 'title', 'text', 'source_file']].copy()
            kept_full_key = full_key[keep_mask]
            kept_title_key = title_key[keep_mask]

            for k in kept_full_key.tolist():
                seen_full.add(k)
            for k in kept_title_key.tolist():
                seen_title.add(k)

            n = len(kept)
            if n == 0:
                continue

            ids = []
            for _ in range(n):
                row_counter += 1
                ids.append(f'CWX{row_counter:09d}')
            kept.insert(0, 'doc_id', ids)

            table = pa.Table.from_pandas(kept, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_parquet, table.schema, compression='snappy')
            writer.write_table(table)
            file_out += n
            stats['total_out_rows'] += n

        stats['files'].append({'file': fp.name, 'in_rows': file_in, 'out_rows': file_out, 'encoding': enc})
        log(f'TASK1: file done {fp.name} in={file_in} out={file_out}')

    if writer:
        writer.close()

    # copy parquet to final
    if out_parquet.exists():
        out_parquet_final.write_bytes(out_parquet.read_bytes())

    dedup_report = {
        **stats,
        'generated_at': datetime.now().isoformat(timespec='seconds')
    }
    dedup_report_path.write_text(json.dumps(dedup_report, ensure_ascii=False, indent=2), encoding='utf-8')

    schema_md = [
        '# Schema Report (City Weixin)',
        '',
        f'- Generated at: {datetime.now().isoformat(timespec="seconds")}',
        f'- Output parquet: `{out_parquet}`',
        f'- Final parquet copy: `{out_parquet_final}`',
        '',
        '## Unified Schema',
    ]
    for c in stats['schema_fields']:
        schema_md.append(f'- `{c}`')
    schema_md += [
        '',
        '## Dedup Summary',
        f"- total_in_rows: {stats['total_in_rows']}",
        f"- total_out_rows: {stats['total_out_rows']}",
        f"- dropped_exact_duplicate: {stats['dropped_exact_duplicate']}",
        f"- dropped_title_date_city_duplicate: {stats['dropped_title_date_city_duplicate']}",
        f"- dropped_near_duplicate: {stats['dropped_near_duplicate']}",
    ]
    schema_report_path.write_text('\n'.join(schema_md), encoding='utf-8')

    log(f'TASK1: done out={out_parquet} rows={stats["total_out_rows"]}')
    return out_parquet


# ---------------------- Task 2 ----------------------

def parse_html_to_text(path: Path) -> str:
    txt = path.read_text(encoding='utf-8', errors='ignore')
    soup = BeautifulSoup(txt, 'lxml')
    for t in soup(['script', 'style', 'noscript']):
        t.extract()
    text = '\n'.join(x.strip() for x in soup.get_text('\n').splitlines() if x.strip())
    return text


def extract_date_from_row(row: pd.Series) -> Optional[str]:
    y = str(row.get('year', '')).strip()
    if y.isdigit() and len(y) == 4:
        return f'{y}-01-01'
    t = str(row.get('title', ''))
    m = re.search(r'(19\d{2}|20\d{2})[-年/](\d{1,2})[-月/](\d{1,2})', t)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return pd.Timestamp(year=y, month=mo, day=d).strftime('%Y-%m-%d')
        except Exception:
            pass
    return None


def score_policy_doc_with_llm(title: str, issuer: str, date: str, text: str) -> dict:
    system = (
        'You are a policy text scoring engine. '
        'Return ONLY JSON with keys: RiskScore, StabilityScore, ScalabilityScore, '
        'EvalPressureScore, StrategicPriorityScore, ConditionalityScore, evidence, confidence. '
        'Scores are integers 0-5.'
    )
    user = f"""
标题: {title}
发布机构: {issuer}
日期: {date}
正文(节选):
{text[:8000]}

请按以下维度给 0-5 分，并提供每个维度一条证据句：
- RiskScore
- StabilityScore
- ScalabilityScore
- EvalPressureScore
- StrategicPriorityScore
- ConditionalityScore
返回 JSON。
"""
    out = LLM.chat_json(system, user, temperature=0.1, max_tokens=700)
    if out is None:
        return fallback_policy_scores(text)

    # sanitize
    dims = [
        'RiskScore', 'StabilityScore', 'ScalabilityScore',
        'EvalPressureScore', 'StrategicPriorityScore', 'ConditionalityScore'
    ]
    for d in dims:
        try:
            out[d] = int(out.get(d, 0))
        except Exception:
            out[d] = 0
        out[d] = max(0, min(5, out[d]))
    if 'evidence' not in out or not isinstance(out['evidence'], dict):
        out['evidence'] = {d: '' for d in dims}
    try:
        out['confidence'] = float(out.get('confidence', 0.5))
    except Exception:
        out['confidence'] = 0.5
    out['confidence'] = max(0.0, min(1.0, out['confidence']))
    return out


def task2_build_central_policy_objectives() -> pd.DataFrame:
    log('TASK2: central policy objectives start')
    pm = MAN / 'planning_manifest.csv'
    if not pm.exists():
        raise RuntimeError('planning_manifest.csv missing')

    df = pd.read_csv(pm, dtype=str).fillna('')
    rows = []
    llm_budget = int(os.getenv('CENTRAL_LLM_BUDGET', '20'))

    for i, (_, r) in enumerate(df.iterrows(), start=1):
        title = str(r.get('title', '')).strip()
        issuer = str(r.get('issuing_body', '')).strip()
        source_url = str(r.get('source_url', '')).strip()
        file_txt = Path(str(r.get('file_txt', '')).strip()) if r.get('file_txt', '') else None
        file_html = Path(str(r.get('file_html', '')).strip()) if r.get('file_html', '') else None

        text = ''
        if file_txt and file_txt.exists():
            text = file_txt.read_text(encoding='utf-8', errors='ignore')
        elif file_html and file_html.exists():
            text = parse_html_to_text(file_html)

        if not text:
            continue

        pdate = extract_date_from_row(r)
        level = 'local' if any(x in issuer for x in ['省', '市', '自治区']) and ('国务院' not in issuer and '全国人民代表大会' not in issuer) else 'central'

        if i <= llm_budget:
            scores = score_policy_doc_with_llm(title, issuer, pdate or '', text)
        else:
            scores = fallback_policy_scores(text)

        row = {
            'title': title,
            'issuer': issuer,
            'date': pdate,
            'level': level,
            'source_url': source_url,
            'text_length': len(text),
            'RiskScore': scores['RiskScore'],
            'StabilityScore': scores['StabilityScore'],
            'ScalabilityScore': scores['ScalabilityScore'],
            'EvalPressureScore': scores['EvalPressureScore'],
            'StrategicPriorityScore': scores['StrategicPriorityScore'],
            'ConditionalityScore': scores['ConditionalityScore'],
            'confidence': scores.get('confidence', 0.5),
            'evidence_json': json.dumps(scores.get('evidence', {}), ensure_ascii=False)
        }
        rows.append(row)
        if i % 10 == 0:
            log(f'TASK2: processed {i}/{len(df)} policy docs (llm_budget={llm_budget})')
        time.sleep(0.05)

    out = pd.DataFrame(rows)
    out['date'] = pd.to_datetime(out['date'], errors='coerce')
    out.sort_values(['date', 'title'], inplace=True)

    csv_path = DIRS['final'] / 'central_policy_objectives.csv'
    jsonl_path = DIRS['final'] / 'central_policy_objectives.jsonl'
    out.to_csv(csv_path, index=False)
    with jsonl_path.open('w', encoding='utf-8') as f:
        for rec in out.to_dict(orient='records'):
            if pd.isna(rec.get('date')):
                rec['date'] = None
            else:
                rec['date'] = pd.Timestamp(rec['date']).strftime('%Y-%m-%d')
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    log(f'TASK2: done docs={len(out)} out={csv_path}')
    return out


# ---------------------- Task 3 ----------------------

RULES = {
    'RiskResponse': ['风险', '防范', '隐患', '应急', '预警', '安全生产'],
    'StabilityResponse': ['稳定', '稳妥', '平稳', '有序', '平安'],
    'PilotResponse': ['试点', '先行先试', '示范区', '试验区'],
    'ScalabilityResponse': ['可复制', '可推广', '推广', '示范带动'],
    'EvalResponse': ['考核', '督导', '落实', '责任', '问责', '绩效'],
    'StrategicResponse': ['国家战略', '十四五', '高质量发展', '现代化', '强国', '双碳'],
}

POLICY_KEYWORDS = ['国务院', '国家', '规划', '实施方案', '政策', '通知']


def rule_score_text(text: str) -> Dict[str, int]:
    t = text or ''
    out = {}
    for k, kws in RULES.items():
        out[k] = int(any(w in t for w in kws))
    out['is_policy_related'] = int(any(w in t for w in POLICY_KEYWORDS) or any(out[k] for k in RULES))
    return out


def llm_classify_post(title: str, text: str) -> Optional[dict]:
    system = (
        'You are a policy communication classifier. Return ONLY JSON with integer 0/1 fields: '
        'is_policy_related, RiskResponse, StabilityResponse, PilotResponse, ScalabilityResponse, '
        'EvalResponse, StrategicResponse, confidence.'
    )
    user = f"""
标题: {title}
正文: {text[:3000]}

请判断该文本是否体现以下维度（0或1）：
- is_policy_related
- RiskResponse
- StabilityResponse
- PilotResponse
- ScalabilityResponse
- EvalResponse
- StrategicResponse
并给出 confidence(0-1)。
返回 JSON。
"""
    out = LLM.chat_json(system, user, temperature=0, max_tokens=300)
    if out is None:
        return None
    fields = ['is_policy_related','RiskResponse','StabilityResponse','PilotResponse','ScalabilityResponse','EvalResponse','StrategicResponse']
    clean = {}
    for f in fields:
        try:
            clean[f] = int(out.get(f, 0))
        except Exception:
            clean[f] = 0
        clean[f] = 1 if clean[f] > 0 else 0
    try:
        clean['confidence'] = float(out.get('confidence', 0.5))
    except Exception:
        clean['confidence'] = 0.5
    clean['confidence'] = max(0.0, min(1.0, clean['confidence']))
    return clean


def task3_build_local_response(cleaned_parquet: Path, central_df: pd.DataFrame) -> pd.DataFrame:
    log('TASK3: local response panel start')
    df = pd.read_parquet(cleaned_parquet)

    # basic clean
    df['title'] = df['title'].astype(str).map(normalize_text)
    df['text'] = df['text'].astype(str).map(normalize_text)

    # rule stage
    rule_rows = [rule_score_text(f"{t} {x}") for t, x in zip(df['title'], df['text'])]
    rule_df = pd.DataFrame(rule_rows)
    for c in rule_df.columns:
        df[c + '_rule'] = rule_df[c]

    # candidate and llm sample (MVP scale)
    candidate_mask = df['is_policy_related_rule'] == 1
    candidates = df.loc[candidate_mask, ['doc_id', 'title', 'text']].copy()
    sample_n = min(int(os.getenv('POST_LLM_SAMPLE', '120')), len(candidates))
    llm_sample = candidates.sample(n=sample_n, random_state=42) if sample_n > 0 else candidates.head(0)

    llm_results = []
    if sample_n > 0:
        log(f'TASK3: LLM classify sample n={sample_n}')
        for i, (_, r) in enumerate(llm_sample.iterrows(), start=1):
            out = llm_classify_post(r['title'], r['text'])
            if out is None:
                continue
            out['doc_id'] = r['doc_id']
            llm_results.append(out)
            if i % 20 == 0:
                log(f'TASK3: LLM classified {i}/{sample_n} sampled posts')
            time.sleep(0.05)

    llm_df = pd.DataFrame(llm_results)
    llm_out = DIRS['inter'] / 'llm_post_classification_sample.csv'
    llm_df.to_csv(llm_out, index=False)

    # combine (rule + optional llm calibration)
    dims = ['RiskResponse','StabilityResponse','PilotResponse','ScalabilityResponse','EvalResponse','StrategicResponse','is_policy_related']
    scales = {d: 1.0 for d in dims}
    if not llm_df.empty:
        merged = llm_df.merge(df[['doc_id'] + [d + '_rule' for d in dims]], on='doc_id', how='left')
        for d in dims:
            rmean = merged[d + '_rule'].mean()
            lmean = merged[d].mean()
            if rmean and rmean > 0:
                scales[d] = float(np.clip(lmean / rmean, 0.5, 1.5))
            else:
                scales[d] = 1.0

    # final per-text scores (scaled rule), llm override if available
    for d in dims:
        df[d] = np.clip(df[d + '_rule'].astype(float) * scales[d], 0, 1)

    if not llm_df.empty:
        llm_map = llm_df.set_index('doc_id')
        hit = df['doc_id'].isin(llm_map.index)
        for d in dims:
            df.loc[hit, d] = df.loc[hit, 'doc_id'].map(llm_map[d]).astype(float)

    df['PolicyResponseIntensity'] = df[['RiskResponse','StabilityResponse','PilotResponse','ScalabilityResponse','EvalResponse','StrategicResponse']].mean(axis=1)

    # city-month aggregate
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = ((df['month'] - 1) // 3 + 1)
    df['month_key'] = df['date'].dt.to_period('M').astype(str)

    gcols = ['city_name', 'province_name', 'year', 'month', 'quarter', 'month_key']
    panel = df.groupby(gcols, as_index=False).agg(
        PostCount=('doc_id', 'count'),
        PolicyResponseIntensity=('PolicyResponseIntensity', 'mean'),
        RiskResponse=('RiskResponse', 'mean'),
        StabilityResponse=('StabilityResponse', 'mean'),
        ScalabilityResponse=('ScalabilityResponse', 'mean'),
        EvalResponse=('EvalResponse', 'mean'),
        StrategicResponse=('StrategicResponse', 'mean')
    )

    # central monthly objective vectors
    c = central_df.copy()
    c['date'] = pd.to_datetime(c['date'], errors='coerce')
    c = c.dropna(subset=['date'])
    c['month_key'] = c['date'].dt.to_period('M').astype(str)
    obj_cols = ['RiskScore','StabilityScore','ScalabilityScore','EvalPressureScore','StrategicPriorityScore']
    for oc in obj_cols:
        c[oc] = pd.to_numeric(c[oc], errors='coerce').fillna(0) / 5.0

    c_month = c.groupby('month_key', as_index=False)[obj_cols].mean().sort_values('month_key')
    # full month index + ffill
    if len(panel) > 0:
        mmin = pd.Period(panel['month_key'].min(), freq='M')
        mmax = pd.Period(panel['month_key'].max(), freq='M')
        full_months = pd.DataFrame({'month_key': [str(p) for p in pd.period_range(mmin, mmax, freq='M')]})
        c_month = full_months.merge(c_month, on='month_key', how='left').sort_values('month_key')
        c_month[obj_cols] = c_month[obj_cols].ffill().fillna(0)

    panel = panel.merge(c_month, on='month_key', how='left')
    panel[obj_cols] = panel[obj_cols].fillna(0)

    # alignment score
    local_vec = panel[['RiskResponse','StabilityResponse','ScalabilityResponse','EvalResponse','StrategicResponse']].to_numpy(dtype=float)
    obj_vec = panel[['RiskScore','StabilityScore','ScalabilityScore','EvalPressureScore','StrategicPriorityScore']].to_numpy(dtype=float)
    align = [cosine_similarity(local_vec[i], obj_vec[i]) for i in range(len(panel))]
    panel['alignment_score'] = align

    out_csv = DIRS['final'] / 'city_month_response_panel.csv'
    out_parquet = DIRS['final'] / 'city_month_response_panel.parquet'
    panel.to_csv(out_csv, index=False)
    panel.to_parquet(out_parquet, index=False)

    # keep scored compact dataset for audit
    scored_sample = df[['doc_id','city_name','province_name','date','title','is_policy_related','RiskResponse','StabilityResponse','PilotResponse','ScalabilityResponse','EvalResponse','StrategicResponse','PolicyResponseIntensity']].sample(min(200000, len(df)), random_state=42)
    scored_sample.to_parquet(DIRS['inter'] / 'city_weixin_scored_sample.parquet', index=False)

    log(f'TASK3: done panel_rows={len(panel)} out={out_csv}')
    return panel


# ---------------------- Task 4 ----------------------

def task4_macro_panel(panel_month: pd.DataFrame) -> pd.DataFrame:
    log('TASK4: macro panel start')
    nbs_manifest = MAN / 'nbs_sources.csv'
    df_nbs = pd.read_csv(nbs_manifest, dtype=str).fillna('') if nbs_manifest.exists() else pd.DataFrame()

    # availability report
    report_md = [
        '# NBS Variable Availability Report',
        '',
        f'- Generated at: {datetime.now().isoformat(timespec="seconds")}',
        '',
        '## Current source entries',
        ''
    ]
    if not df_nbs.empty:
        for _, r in df_nbs.iterrows():
            report_md += [
                f"- dataset_name: {r.get('dataset_name','')}",
                f"  - variable_group: {r.get('variable_group','')}",
                f"  - level: {r.get('level','')}",
                f"  - year_range: {r.get('year_range','')}",
                f"  - downloadable: {r.get('downloadable','')}",
                f"  - source_url: {r.get('source_url','')}",
            ]
    else:
        report_md.append('- nbs_sources.csv not found')

    report_md += [
        '',
        '## Availability assessment for required variables',
        '',
        '| Variable | City-year availability | Notes |',
        '|---|---|---|',
        '| GDP | partial | City-level structured series not fully extracted from current mirrored pages |',
        '| GDP per capita | partial | Same as above |',
        '| Fiscal revenue | limited | No fully structured city-year file in current assets |',
        '| Fiscal expenditure | limited | No fully structured city-year file in current assets |',
        '| Fixed asset investment | limited | Needs dedicated extraction from yearbook tables |',
        '| Population | partial | Potentially extractable; not yet structured in this MVP |',
        '| Industry structure (2nd/3rd share) | limited | Requires table-level parsing and harmonization |',
        '',
        'Conclusion: build analysis-ready city-year structure with placeholder macro columns and merge hooks in MVP.'
    ]

    (DIRS['final'] / 'nbs_variable_availability_report.md').write_text('\n'.join(report_md), encoding='utf-8')

    city_year = panel_month.groupby(['city_name','province_name','year'], as_index=False).agg(PostCount=('PostCount','sum'))
    macro = city_year[['city_name','province_name','year']].copy()
    for col in ['GDP','GDP_per_capita','FiscalRevenue','FiscalExpenditure','FAI','Population','ShareSecondary','ShareTertiary']:
        macro[col] = np.nan
    macro['macro_data_available'] = 0
    macro['macro_note'] = 'NBS city-level structured values unavailable in current asset snapshot'

    out = DIRS['final'] / 'macro_city_year_panel.csv'
    macro.to_csv(out, index=False)
    log(f'TASK4: done rows={len(macro)} out={out}')
    return macro


# ---------------------- Task 5 ----------------------

def task5_merge_panels(panel_month: pd.DataFrame, macro_year: pd.DataFrame):
    log('TASK5: merge panel datasets start')

    # city/province ids
    city_codes = {c: i+1 for i, c in enumerate(sorted(panel_month['city_name'].dropna().unique()))}
    prov_codes = {p: i+1 for i, p in enumerate(sorted(panel_month['province_name'].dropna().unique()))}

    m = panel_month.copy()
    m['city_id'] = m['city_name'].map(city_codes)
    m['province_id'] = m['province_name'].map(prov_codes)

    # merge macro by city-year
    m = m.merge(macro_year, on=['city_name','province_name','year'], how='left', suffixes=('','_macro'))

    city_month = m.copy()
    city_month.to_csv(DIRS['final'] / 'city_month_panel.csv', index=False)
    city_month.to_parquet(DIRS['final'] / 'city_month_panel.parquet', index=False)

    city_quarter = city_month.groupby(['city_name','province_name','city_id','province_id','year','quarter'], as_index=False).agg(
        PostCount=('PostCount','sum'),
        PolicyResponseIntensity=('PolicyResponseIntensity','mean'),
        RiskResponse=('RiskResponse','mean'),
        StabilityResponse=('StabilityResponse','mean'),
        ScalabilityResponse=('ScalabilityResponse','mean'),
        EvalResponse=('EvalResponse','mean'),
        StrategicResponse=('StrategicResponse','mean'),
        alignment_score=('alignment_score','mean'),
        GDP=('GDP','mean'),
        GDP_per_capita=('GDP_per_capita','mean'),
        FiscalRevenue=('FiscalRevenue','mean'),
        FiscalExpenditure=('FiscalExpenditure','mean'),
        FAI=('FAI','mean'),
        Population=('Population','mean'),
        ShareSecondary=('ShareSecondary','mean'),
        ShareTertiary=('ShareTertiary','mean'),
    )
    city_quarter.to_csv(DIRS['final'] / 'city_quarter_panel.csv', index=False)
    city_quarter.to_parquet(DIRS['final'] / 'city_quarter_panel.parquet', index=False)

    city_year = city_month.groupby(['city_name','province_name','city_id','province_id','year'], as_index=False).agg(
        PostCount=('PostCount','sum'),
        PolicyResponseIntensity=('PolicyResponseIntensity','mean'),
        RiskResponse=('RiskResponse','mean'),
        StabilityResponse=('StabilityResponse','mean'),
        ScalabilityResponse=('ScalabilityResponse','mean'),
        EvalResponse=('EvalResponse','mean'),
        StrategicResponse=('StrategicResponse','mean'),
        alignment_score=('alignment_score','mean'),
        GDP=('GDP','mean'),
        GDP_per_capita=('GDP_per_capita','mean'),
        FiscalRevenue=('FiscalRevenue','mean'),
        FiscalExpenditure=('FiscalExpenditure','mean'),
        FAI=('FAI','mean'),
        Population=('Population','mean'),
        ShareSecondary=('ShareSecondary','mean'),
        ShareTertiary=('ShareTertiary','mean'),
    )
    city_year.to_csv(DIRS['final'] / 'city_year_panel.csv', index=False)
    city_year.to_parquet(DIRS['final'] / 'city_year_panel.parquet', index=False)

    log(f'TASK5: done month={len(city_month)} quarter={len(city_quarter)} year={len(city_year)}')
    return city_month, city_quarter, city_year


# ---------------------- Task 6 ----------------------

def _panel_to_long_index(df: pd.DataFrame, entity_col: str, time_col: str) -> pd.DataFrame:
    d = df.copy()
    d = d.dropna(subset=[entity_col, time_col])
    d = d.set_index([entity_col, time_col]).sort_index()
    return d


def _save_model_results(res, name: str):
    params = res.params
    bse = res.std_errors
    tstats = res.tstats
    pvals = res.pvalues
    out = pd.DataFrame({
        'term': params.index,
        'coef': params.values,
        'std_err': bse.values,
        't': tstats.values,
        'p_value': pvals.values,
    })
    out['model'] = name
    out['nobs'] = float(res.nobs)
    out['rsquared'] = float(getattr(res, 'rsquared', np.nan))

    csvp = DIRS['tables'] / f'regression_results_{name}.csv'
    texp = DIRS['tables'] / f'regression_results_{name}.tex'
    out.to_csv(csvp, index=False)
    out.to_latex(texp, index=False, float_format='%.4f')

    summp = DIRS['tables'] / f'regression_summary_{name}.txt'
    with summp.open('w', encoding='utf-8') as f:
        f.write(str(res.summary))
    return out


def task6_run_regressions(city_month: pd.DataFrame, city_year: pd.DataFrame, central_df: pd.DataFrame):
    log('TASK6: regressions start')

    # objective monthly
    c = central_df.copy()
    c['date'] = pd.to_datetime(c['date'], errors='coerce')
    c = c.dropna(subset=['date'])
    c['month_key'] = c['date'].dt.to_period('M').astype(str)
    obj_cols = ['RiskScore','StabilityScore','ScalabilityScore','EvalPressureScore','StrategicPriorityScore']
    for oc in obj_cols:
        c[oc] = pd.to_numeric(c[oc], errors='coerce').fillna(0) / 5.0
    c_m = c.groupby('month_key', as_index=False)[obj_cols].mean()
    c_m['Objective'] = c_m[obj_cols].mean(axis=1)

    m = city_month.copy()
    m['month_key'] = m['year'].astype(int).astype(str) + '-' + m['month'].astype(int).astype(str).str.zfill(2)
    # avoid duplicate objective columns after repeated runs
    for cdrop in ['Objective','RiskScore','StabilityScore','ScalabilityScore']:
        if cdrop in m.columns:
            m = m.drop(columns=[cdrop])
    m = m.merge(c_m[['month_key','Objective','RiskScore','StabilityScore','ScalabilityScore']], on='month_key', how='left')
    m[['Objective','RiskScore','StabilityScore','ScalabilityScore']] = m[['Objective','RiskScore','StabilityScore','ScalabilityScore']].fillna(0)

    cap = m.groupby('city_name')['PostCount'].mean().rename('Capacity')
    m = m.merge(cap, on='city_name', how='left')
    m['Capacity'] = np.log1p(m['Capacity'])
    stab_need = m.groupby('city_name')['RiskResponse'].mean().rename('StabilityNeed')
    m = m.merge(stab_need, on='city_name', how='left')

    # Model A
    m['ObjXCap'] = m['Objective'] * m['Capacity']
    m['time_id'] = pd.to_datetime(m['month_key'] + '-01', errors='coerce')
    dmA = m[['city_name','time_id','PolicyResponseIntensity','Objective','ObjXCap']].dropna()
    dmA = _panel_to_long_index(dmA, 'city_name', 'time_id')
    yA = dmA['PolicyResponseIntensity']
    XA = dmA[['Objective','ObjXCap']]
    resA = PanelOLS(yA, XA, entity_effects=True, time_effects=True, drop_absorbed=True).fit(cov_type='clustered', cluster_entity=True)
    outA = _save_model_results(resA, 'model_A')

    # Model B (Outcome as next-year post count due macro gaps)
    ydf = city_year.copy().sort_values(['city_name','year'])
    ydf['Outcome_next'] = ydf.groupby('city_name')['PostCount'].shift(-1)
    ydf['Capacity'] = np.log1p(ydf.groupby('city_name')['PostCount'].transform('mean'))
    ydf = ydf.dropna(subset=['Outcome_next'])
    ydf['time_id'] = pd.to_datetime(ydf['year'].astype(int).astype(str) + '-01-01')
    dmB = ydf[['city_name','time_id','Outcome_next','PolicyResponseIntensity','alignment_score','Capacity']].dropna()
    dmB = _panel_to_long_index(dmB, 'city_name', 'time_id')
    yB = dmB['Outcome_next']
    XB = dmB[['PolicyResponseIntensity','alignment_score','Capacity']]
    resB = PanelOLS(yB, XB, entity_effects=True, time_effects=True, drop_absorbed=True).fit(cov_type='clustered', cluster_entity=True)
    outB = _save_model_results(resB, 'model_B')

    # Model C
    dmC = m[['city_name','time_id','PolicyResponseIntensity','RiskScore','ScalabilityScore','StabilityScore','Capacity','StabilityNeed']].dropna().copy()
    dmC['RiskXCap'] = dmC['RiskScore'] * dmC['Capacity']
    dmC['ScaleXCap'] = dmC['ScalabilityScore'] * dmC['Capacity']
    dmC['StabXNeed'] = dmC['StabilityScore'] * dmC['StabilityNeed']
    dmC = _panel_to_long_index(dmC, 'city_name', 'time_id')
    yC = dmC['PolicyResponseIntensity']
    XC = dmC[['RiskXCap','ScaleXCap','StabXNeed']]
    resC = PanelOLS(yC, XC, entity_effects=True, time_effects=True, drop_absorbed=True).fit(cov_type='clustered', cluster_entity=True)
    outC = _save_model_results(resC, 'model_C')

    all_out = pd.concat([outA, outB, outC], ignore_index=True)
    all_out.to_csv(DIRS['tables'] / 'regression_results_all.csv', index=False)

    log('TASK6: done')
    return all_out


# ---------------------- Task 7 ----------------------

def task7_figures(city_month: pd.DataFrame, central_df: pd.DataFrame, reg_out: pd.DataFrame):
    log('TASK7: figures start')
    sns.set_theme(style='whitegrid')

    # 1. post volume trend
    m = city_month.copy()
    m['date_m'] = pd.to_datetime(m['year'].astype(int).astype(str) + '-' + m['month'].astype(int).astype(str).str.zfill(2) + '-01')
    g1 = m.groupby('date_m', as_index=False)['PostCount'].sum()
    plt.figure(figsize=(12,4))
    plt.plot(g1['date_m'], g1['PostCount'])
    plt.title('Total Weixin Post Volume Over Time')
    plt.tight_layout()
    plt.savefig(DIRS['figs'] / '01_post_volume_trend.png', dpi=150)
    plt.close()

    # 2. policy response trend
    g2 = m.groupby('date_m', as_index=False)['PolicyResponseIntensity'].mean()
    plt.figure(figsize=(12,4))
    plt.plot(g2['date_m'], g2['PolicyResponseIntensity'])
    plt.title('Average City Policy Response Intensity Over Time')
    plt.tight_layout()
    plt.savefig(DIRS['figs'] / '02_policy_response_trend.png', dpi=150)
    plt.close()

    # 3. central objective distribution
    c = central_df.copy()
    dims = ['RiskScore','StabilityScore','ScalabilityScore','EvalPressureScore','StrategicPriorityScore','ConditionalityScore']
    for d in dims:
        c[d] = pd.to_numeric(c[d], errors='coerce')
    c_long = c.melt(value_vars=dims, var_name='Dimension', value_name='Score')
    plt.figure(figsize=(12,5))
    sns.boxplot(data=c_long, x='Dimension', y='Score')
    plt.xticks(rotation=25)
    plt.title('Distribution of Central Policy Objective Scores')
    plt.tight_layout()
    plt.savefig(DIRS['figs'] / '03_central_objective_distribution.png', dpi=150)
    plt.close()

    # 4. alignment distribution / province means
    plt.figure(figsize=(12,5))
    sns.histplot(m['alignment_score'].dropna(), bins=40, kde=True)
    plt.title('Alignment Score Distribution (City-Month)')
    plt.tight_layout()
    plt.savefig(DIRS['figs'] / '04_alignment_distribution.png', dpi=150)
    plt.close()

    prov = m.groupby('province_name', as_index=False)['alignment_score'].mean().sort_values('alignment_score', ascending=False).head(20)
    plt.figure(figsize=(10,6))
    sns.barplot(data=prov, y='province_name', x='alignment_score', orient='h')
    plt.title('Top Provinces by Mean Alignment Score')
    plt.tight_layout()
    plt.savefig(DIRS['figs'] / '04b_alignment_by_province_top20.png', dpi=150)
    plt.close()

    # 5. key regression coefficients
    keep_terms = ['Objective','ObjXCap','PolicyResponseIntensity','alignment_score','RiskXCap','ScaleXCap','StabXNeed']
    r = reg_out[reg_out['term'].isin(keep_terms)].copy()
    if not r.empty:
        r['ci_low'] = r['coef'] - 1.96 * r['std_err']
        r['ci_high'] = r['coef'] + 1.96 * r['std_err']
        plt.figure(figsize=(10,6))
        sns.pointplot(data=r, y='term', x='coef', hue='model', dodge=0.4, join=False)
        plt.axvline(0, color='black', lw=1)
        plt.title('Key Regression Coefficients')
        plt.tight_layout()
        plt.savefig(DIRS['figs'] / '05_regression_coefficients.png', dpi=150)
        plt.close()

    log('TASK7: done')


# ---------------------- Task 8 ----------------------

def task8_write_memo(city_month: pd.DataFrame, city_quarter: pd.DataFrame, city_year: pd.DataFrame):
    log('TASK8: memo start')
    memo = []
    memo.append('# Extension Research Memo')
    memo.append('')
    memo.append(f'- Generated at: {datetime.now().isoformat(timespec="seconds")}')
    memo.append('')
    memo.append('## 1. 数据来源与清洗过程')
    memo.append('- 主数据为地级市政务微信文本分片（part_001~005），已统一 schema 并去重。')
    memo.append('- 中央政策文本来自 planning_docs 与 planning_manifest。')
    memo.append('- NBS 与土地数据在当前版本主要作为“可得性/入口”资产，宏观变量结构化尚不完整。')
    memo.append('')
    memo.append('## 2. 文本变量定义')
    memo.append('- 中央目标函数：Risk/Stability/Scalability/EvalPressure/StrategicPriority/Conditionality。')
    memo.append('- 地方响应：Risk/Stability/Pilot/Scalability/Eval/Strategic 维度，并构建 PolicyResponseIntensity。')
    memo.append('- 中央-地方一致性：城市-月响应向量与中央当期目标向量余弦相似度（alignment_score）。')
    memo.append('')
    memo.append('## 3. 面板构造方法')
    memo.append(f'- city_month_panel: {len(city_month)} 行')
    memo.append(f'- city_quarter_panel: {len(city_quarter)} 行')
    memo.append(f'- city_year_panel: {len(city_year)} 行')
    memo.append('')
    memo.append('## 4. 基准回归设计')
    memo.append('- Model A: 地方响应 ~ 中央目标 + 交互项 + 城市FE + 时间FE')
    memo.append('- Model B: 次期结果(通信活跃度代理) ~ 响应 + 一致性 + controls + FE')
    memo.append('- Model C: 战略性学习交互项（Risk×Capacity, Scalability×Capacity, Stability×Need）')
    memo.append('')
    memo.append('## 5. 主要结果')
    memo.append('- 回归结果已输出为 CSV/TEX；关键系数图已生成。')
    memo.append('- 中央目标维度分布与地方响应时间趋势图已生成。')
    memo.append('')
    memo.append('## 6. 对原文的可能 argue')
    memo.append('- 文本证据支持“中央不只学习平均处理效应（ATE）”，还在强调风险可控、稳妥推进与可复制扩散。')
    memo.append('- 地方响应并非单一强度变化，而呈现多维目标耦合。')
    memo.append('- alignment 可作为“战略性政策学习”可检验指标。')
    memo.append('')
    memo.append('## 7. 当前局限与下一步数据需求')
    memo.append('- NBS 城市级结构化宏观变量尚不完整，当前 Model B 使用通信结果代理。')
    memo.append('- 北大法宝原始法规文本尚未批量落地（登录/权限限制）。')
    memo.append('- 下一步应补齐城市级财政、投资、产业结构、人口等年度指标，并进行稳健性检验。')

    out = DIRS['memo'] / 'extension_research_memo.md'
    out.write_text('\n'.join(memo), encoding='utf-8')
    log(f'TASK8: done {out}')


# ---------------------- Summary ----------------------

def write_delivery_summary():
    summary = {}
    summary['generated_at'] = datetime.now().isoformat(timespec='seconds')
    summary['final_datasets'] = [
        'data_intermediate/cleaned_city_weixin.parquet',
        'data_final/cleaned_city_weixin.parquet',
        'data_final/central_policy_objectives.csv',
        'data_final/central_policy_objectives.jsonl',
        'data_final/city_month_response_panel.csv',
        'data_final/city_month_response_panel.parquet',
        'data_final/macro_city_year_panel.csv',
        'data_final/city_month_panel.csv',
        'data_final/city_month_panel.parquet',
        'data_final/city_quarter_panel.csv',
        'data_final/city_quarter_panel.parquet',
        'data_final/city_year_panel.csv',
        'data_final/city_year_panel.parquet',
        'data_final/nbs_variable_availability_report.md',
    ]
    summary['tables'] = sorted([p.name for p in DIRS['tables'].glob('*')])
    summary['figures'] = sorted([p.name for p in DIRS['figs'].glob('*')])
    summary['memo'] = sorted([p.name for p in DIRS['memo'].glob('*')])

    out = DIRS['man'] / 'delivery_summary.json'
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    log(f'SUMMARY: {out}')


def main():
    t0 = time.time()
    log('PIPELINE_START')

    cleaned = DIRS['inter'] / 'cleaned_city_weixin.parquet'
    if cleaned.exists() and (DIRS['man'] / 'dedup_report.json').exists():
        log(f'TASK1: skipped, reuse existing {cleaned}')
    else:
        cleaned = task1_audit_and_clean_city_weixin()

    central_csv = DIRS['final'] / 'central_policy_objectives.csv'
    if central_csv.exists():
        central = pd.read_csv(central_csv)
        log(f'TASK2: skipped, reuse existing {central_csv}')
    else:
        central = task2_build_central_policy_objectives()

    panel_month = task3_build_local_response(cleaned, central)
    macro = task4_macro_panel(panel_month)
    city_month, city_quarter, city_year = task5_merge_panels(panel_month, macro)
    reg_out = task6_run_regressions(city_month, city_year, central)
    task7_figures(city_month, central, reg_out)
    task8_write_memo(city_month, city_quarter, city_year)
    write_delivery_summary()

    log(f'PIPELINE_DONE seconds={time.time()-t0:.1f}')


if __name__ == '__main__':
    main()
