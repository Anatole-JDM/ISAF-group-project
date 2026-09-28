import zipfile, io, numpy as np, pandas as pd
import pyarrow as pa, pyarrow.compute as pc
from pyarrow import csv as pacsv
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
BASE='C:/Users/smaye/OneDrive/Bureau/X-HEC/HEC_courses/algo_fairness/ISAF_gp/ISAF-group-project/opp_data/'
def T(c): return pc.fill_null(pc.equal(pc.utf8_lower(c),'true'),False)

def load(path, cols, consent_only=True):
    z=zipfile.ZipFile(path); info=z.infolist()[0]
    with z.open(info) as f: have=next(io.TextIOWrapper(f,'utf8',errors='replace')).strip().split(',')
    use=[c for c in cols if c in have]
    parts=[]
    with z.open(info) as f:
        rdr=pacsv.open_csv(f, read_options=pacsv.ReadOptions(block_size=1<<26),
            convert_options=pacsv.ConvertOptions(include_columns=use, column_types={c:pa.string() for c in use}))
        for b in rdr:
            t=b.filter(T(b.column('search_conducted')))
            if t.num_rows==0: continue
            if consent_only: t=t.filter(pc.fill_null(pc.equal(t.column('search_basis'),'consent'),False))
            if t.num_rows: parts.append(t.to_pandas())
    return pd.concat(parts, ignore_index=True)

def prep(df, cat, num):
    df=df.copy()
    df['y']=df['contraband_found'].str.lower().eq('true').astype(int)
    d=pd.to_datetime(df['date'], errors='coerce')
    df['year']=d.dt.year; df['month']=d.dt.month; df['dow']=d.dt.dayofweek
    if 'time' in df: df['hour']=pd.to_numeric(df['time'].str.slice(0,2), errors='coerce')
    if 'subject_age' in df: df['age']=pd.to_numeric(df['subject_age'], errors='coerce')
    for c in ['lat','lng']:
        if c in df: df[c]=pd.to_numeric(df[c], errors='coerce')
    if 'vehicle_registration_state' in df:
        df['out_of_state']=(df['vehicle_registration_state']!='TN').astype(int)
    df=df.dropna(subset=['year'])
    return df

def run(tag, df, cat, num, split_year):
    df=df[df['year'].notna()]
    tr=df[df['year']<split_year]; te=df[df['year']>=split_year]
    X=[c for c in cat+num if c in df.columns]
    pre=ColumnTransformer([
        ('c', Pipeline([('i',SimpleImputer(strategy='constant',fill_value='MISSING')),
                        ('o',OneHotEncoder(handle_unknown='ignore',min_frequency=30))]), [c for c in cat if c in df.columns]),
        ('n', Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler())]), [c for c in num if c in df.columns])])
    print(f"\n### {tag}: train {len(tr):,} (<{split_year}) / test {len(te):,} (>={split_year}); base rate train {tr.y.mean():.1%} test {te.y.mean():.1%}")
    lr=Pipeline([('p',pre),('m',LogisticRegression(max_iter=2000,C=1.0))]).fit(tr[X],tr.y)
    p_lr=lr.predict_proba(te[X])[:,1]
    print(f"  Logistic  AUC = {roc_auc_score(te.y,p_lr):.4f}")
    Xb=tr[X].copy(); Xt=te[X].copy()
    for c in [c for c in cat if c in df.columns]:
        Xb[c]=Xb[c].astype('category'); Xt[c]=pd.Categorical(Xt[c], categories=Xb[c].cat.categories)
    gb=HistGradientBoostingClassifier(max_iter=300, categorical_features=[c for c in cat if c in df.columns], random_state=0).fit(Xb,tr.y)
    p_gb=gb.predict_proba(Xt)[:,1]
    print(f"  HistGB    AUC = {roc_auc_score(te.y,p_gb):.4f}")
    for g in ['white','black','hispanic']:
        m=te['subject_race']==g
        if m.sum()>500:
            print(f"    {g:10} n={m.sum():>7,}  base={te.y[m].mean():.1%}  AUC_gb={roc_auc_score(te.y[m],p_gb[m]):.4f}")
    return te, p_gb

NASH=['date','time','lat','lng','precinct','zone','reporting_area','subject_age','subject_race','subject_sex',
      'reason_for_stop','vehicle_registration_state','search_basis','contraband_found','search_conducted','officer_id_hash']
NC=['date','time','county_name','department_name','subject_age','subject_race','subject_sex','reason_for_stop',
    'search_basis','contraband_found','search_conducted','officer_id_hash']
print('loading nashville...', flush=True)
n=prep(load(BASE+'tn_nashville_2020_04_01.csv.zip', NASH), None, None)
run('NASHVILLE consent', n, ['subject_sex','reason_for_stop','precinct','zone'], ['age','hour','dow','month','lat','lng','out_of_state'], 2017)
print('\nloading nc...', flush=True)
c=prep(load(BASE+'nc_statewide_2020_04_01.csv.zip', NC), None, None)
c=c[c['year']>=2002]
run('NC consent', c, ['subject_sex','reason_for_stop','county_name','department_name'], ['age','hour','dow','month'], 2013)
