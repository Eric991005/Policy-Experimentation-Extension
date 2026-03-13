#!/usr/bin/env python3
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import folium

BASE = Path('/root/autodl-tmp/policy_data_collection/project_extension')
DATA_FINAL = BASE / 'data_final'
FIG = BASE / 'outputs' / 'figures'
MAN = BASE / 'manifests'
LOG = BASE / 'logs' / 'pipeline.log'
MEMO = BASE / 'outputs' / 'memo' / 'extension_research_memo.md'

for d in [FIG, MAN]:
    d.mkdir(parents=True, exist_ok=True)


def log(msg: str):
    line = f"{datetime.now().isoformat(timespec='seconds')} [CITYMAP] {msg}"
    print(line, flush=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def norm_prov(s: str) -> str:
    s = str(s or '').strip()
    for r in ['壮族自治区', '回族自治区', '维吾尔自治区', '自治区', '特别行政区', '省', '市']:
        s = s.replace(r, '')
    return s


def norm_city(s: str) -> str:
    s = str(s or '').strip()
    repls = ['市辖区', '地区', '自治州', '盟', '林区', '土家族苗族自治州', '藏族自治州', '回族自治州', '朝鲜族自治州', '蒙古族藏族自治州', '布依族苗族自治州', '哈萨克自治州', '傣族自治州', '傈僳族自治州', '白族自治州', '彝族自治州', '壮族苗族自治州', '黎族自治县', '苗族侗族自治州', '藏族羌族自治州']
    for r in repls:
        s = s.replace(r, '')
    if s.endswith('市'):
        s = s[:-1]
    return s.strip()


def get_json(url: str) -> dict:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def build_city_geojson() -> dict:
    country = get_json('https://geo.datav.aliyun.com/areas_v3/bound/100000_full.json')
    direct_city = {'北京市', '天津市', '上海市', '重庆市'}

    features = []
    for pft in country.get('features', []):
        pp = pft.get('properties', {})
        prov_name = pp.get('name', '')
        prov_code = str(pp.get('adcode', ''))
        if prov_name in ['香港特别行政区', '澳门特别行政区']:
            continue

        # 台湾省：即使没有城市级样本指标，也强制保留为“空值占位”面要素
        if prov_name == '台湾省':
            pft2 = dict(pft)
            pft2['properties'] = dict(pp)
            pft2['properties']['province_name'] = prov_name
            pft2['properties']['city_name'] = prov_name
            pft2['properties']['city_adcode'] = str(pp.get('adcode', ''))
            pft2['properties']['level'] = 'province_as_city'
            pft2['properties']['force_empty'] = True
            features.append(pft2)
            continue

        if prov_name in direct_city:
            pft2 = dict(pft)
            pft2['properties'] = dict(pp)
            pft2['properties']['province_name'] = prov_name
            pft2['properties']['city_name'] = prov_name
            pft2['properties']['city_adcode'] = str(pp.get('adcode', ''))
            pft2['properties']['level'] = 'city'
            pft2['properties']['force_empty'] = False
            features.append(pft2)
            continue

        try:
            sub = get_json(f'https://geo.datav.aliyun.com/areas_v3/bound/{prov_code}_full.json')
        except Exception as e:
            log(f'WARN fetch province {prov_name}({prov_code}) failed: {e}')
            continue

        for cft in sub.get('features', []):
            cp = cft.get('properties', {})
            lvl = cp.get('level', '')
            if lvl not in ['city']:
                continue
            cft2 = dict(cft)
            cft2['properties'] = dict(cp)
            cft2['properties']['province_name'] = prov_name
            cft2['properties']['city_name'] = cp.get('name', '')
            cft2['properties']['city_adcode'] = str(cp.get('adcode', ''))
            cft2['properties']['force_empty'] = False
            features.append(cft2)

    geo = {'type': 'FeatureCollection', 'features': features}
    return geo


def main():
    log('start city-level map generation')

    cm = pd.read_csv(DATA_FINAL / 'city_month_panel.csv')
    city = cm.groupby(['province_name', 'city_name'], as_index=False).agg(
        alignment_score=('alignment_score', 'mean'),
        PolicyResponseIntensity=('PolicyResponseIntensity', 'mean'),
        PostCount=('PostCount', 'sum')
    )
    city['prov_norm'] = city['province_name'].map(norm_prov)
    city['city_norm'] = city['city_name'].map(norm_city)

    geo = build_city_geojson()
    log(f'geo city features={len(geo["features"])}')

    rows = []
    for ft in geo['features']:
        p = ft.get('properties', {})
        rows.append({
            'city_adcode': str(p.get('city_adcode', p.get('adcode', ''))),
            'province_name_geo': p.get('province_name', ''),
            'city_name_geo': p.get('city_name', p.get('name', '')),
            'prov_norm': norm_prov(p.get('province_name', '')),
            'city_norm': norm_city(p.get('city_name', p.get('name', ''))),
            'force_empty': bool(p.get('force_empty', False)),
        })
    gdf = pd.DataFrame(rows)

    merged = gdf.merge(city, on=['prov_norm', 'city_norm'], how='left')

    # 强制空值占位（如台湾省）：即使偶然匹配到同名记录，也显示为空
    force_empty_mask = merged['force_empty'].fillna(False)
    merged.loc[force_empty_mask, ['alignment_score', 'PolicyResponseIntensity', 'PostCount']] = np.nan

    matched = merged['alignment_score'].notna().sum()
    match_rate = matched / len(merged) if len(merged) else 0
    log(f'match city metrics matched={matched}/{len(merged)} rate={match_rate:.3f}; force_empty={int(force_empty_mask.sum())}')

    # fill props back
    by_code = {
        str(r['city_adcode']): r for _, r in merged.iterrows()
    }
    for ft in geo['features']:
        p = ft['properties']
        code = str(p.get('city_adcode', p.get('adcode', '')))
        r = by_code.get(code)
        force_empty = bool(p.get('force_empty', False))
        if r is None or force_empty:
            p['alignment_score'] = None
            p['PolicyResponseIntensity'] = None
            p['PostCount'] = None
        else:
            p['alignment_score'] = None if pd.isna(r.get('alignment_score')) else float(r.get('alignment_score'))
            p['PolicyResponseIntensity'] = None if pd.isna(r.get('PolicyResponseIntensity')) else float(r.get('PolicyResponseIntensity'))
            p['PostCount'] = None if pd.isna(r.get('PostCount')) else float(r.get('PostCount'))

    # save merged table + geojson
    merged.to_csv(DATA_FINAL / 'city_geo_metric_merge_status.csv', index=False)
    with (DATA_FINAL / 'city_level_boundary_with_metrics.geojson').open('w', encoding='utf-8') as f:
        json.dump(geo, f, ensure_ascii=False)

    # maps
    value_df = merged[['city_adcode', 'alignment_score', 'PolicyResponseIntensity', 'city_name_geo', 'province_name_geo']].copy()
    value_df['city_adcode'] = value_df['city_adcode'].astype(str)

    m1 = folium.Map(location=[35.9, 104.2], zoom_start=4, tiles='cartodbpositron')
    folium.Choropleth(
        geo_data=geo,
        data=value_df[['city_adcode', 'alignment_score']],
        columns=['city_adcode', 'alignment_score'],
        key_on='feature.properties.city_adcode',
        fill_color='BuGn',
        fill_opacity=0.9,
        line_opacity=0.28,
        legend_name='City Mean Alignment Score',
        nan_fill_color='#A3A3A3',
        nan_fill_opacity=0.98,
        line_color='white',
        line_weight=0.35
    ).add_to(m1)
    folium.GeoJson(
        geo,
        tooltip=folium.GeoJsonTooltip(
            fields=['province_name', 'city_name', 'alignment_score', 'PolicyResponseIntensity', 'force_empty'],
            aliases=['省份', '城市', 'Alignment', 'Response', '空值占位'],
            localize=True
        )
    ).add_to(m1)
    out1 = FIG / '11_city_alignment_map.html'
    m1.save(str(out1))

    m2 = folium.Map(location=[35.9, 104.2], zoom_start=4, tiles='cartodbpositron')
    folium.Choropleth(
        geo_data=geo,
        data=value_df[['city_adcode', 'PolicyResponseIntensity']],
        columns=['city_adcode', 'PolicyResponseIntensity'],
        key_on='feature.properties.city_adcode',
        fill_color='OrRd',
        fill_opacity=0.9,
        line_opacity=0.28,
        legend_name='City Mean Policy Response Intensity',
        nan_fill_color='#A3A3A3',
        nan_fill_opacity=0.98,
        line_color='white',
        line_weight=0.35
    ).add_to(m2)
    folium.GeoJson(
        geo,
        tooltip=folium.GeoJsonTooltip(
            fields=['province_name', 'city_name', 'PolicyResponseIntensity', 'alignment_score', 'force_empty'],
            aliases=['省份', '城市', 'Response', 'Alignment', '空值占位'],
            localize=True
        )
    ).add_to(m2)
    out2 = FIG / '12_city_response_map.html'
    m2.save(str(out2))

    summary = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'city_boundary_features': int(len(merged)),
        'city_metric_matched': int(matched),
        'city_metric_match_rate': float(match_rate),
        'force_empty_features': int(force_empty_mask.sum()),
        'force_empty_regions': merged.loc[force_empty_mask, ['province_name_geo', 'city_name_geo']].drop_duplicates().to_dict(orient='records'),
        'outputs': {
            'map_alignment_html': str(out1),
            'map_response_html': str(out2),
            'merge_status_csv': str(DATA_FINAL / 'city_geo_metric_merge_status.csv'),
            'city_geojson': str(DATA_FINAL / 'city_level_boundary_with_metrics.geojson')
        }
    }
    (MAN / 'city_level_map_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    memo_lines = [
        '\n## 10. 地级市边界地图（新增）',
        f'- 更新时间: {datetime.now().isoformat(timespec="seconds")}',
        '- 数据源: 阿里云 DataV 行政边界（省下辖地级市边界） + city_month_panel 聚合指标。',
        f'- 边界数: {len(merged)}；成功匹配城市指标: {matched}（match rate={match_rate:.1%}）。',
        '- 输出文件:',
        '  - outputs/figures/11_city_alignment_map.html',
        '  - outputs/figures/12_city_response_map.html',
        '  - data_final/city_level_boundary_with_metrics.geojson',
        '  - data_final/city_geo_metric_merge_status.csv',
    ]
    if MEMO.exists():
        with MEMO.open('a', encoding='utf-8') as f:
            f.write('\n'.join(memo_lines) + '\n')

    log('done city-level map generation')


if __name__ == '__main__':
    main()
