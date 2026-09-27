import zipfile, io, json
import pyarrow as pa, pyarrow.compute as pc
from pyarrow import csv as pacsv
BASE='C:/Users/smaye/OneDrive/Bureau/X-HEC/HEC_courses/algo_fairness/ISAF_gp/ISAF-group-project/opp_data/'
JOBS={'tn_nashville':BASE+'tn_nashville_2020_04_01.csv.zip','nc_statewide':BASE+'nc_statewide_2020_04_01.csv.zip'}
def T(col): return pc.fill_null(pc.equal(pc.utf8_lower(col),'true'),False)
for name,path in JOBS.items():
    z=zipfile.ZipFile(path); info=z.infolist()[0]
    with z.open(info) as f: cols=next(io.TextIOWrapper(f,'utf8',errors='replace')).strip().split(',')
    use=[c for c in ['date','search_conducted','contraband_found','contraband_drugs','contraband_weapons','officer_id_hash','search_basis','arrest_made'] if c in cols]
    agg={'n':0,'found':0,'drugs':0,'weap':0,'F_but_drugsweap':0,'T_no_drugsweap':0,'year':{},'off':{}}
    with z.open(info) as f:
        rdr=pacsv.open_csv(f, read_options=pacsv.ReadOptions(block_size=1<<26),
            convert_options=pacsv.ConvertOptions(include_columns=use, column_types={c:pa.string() for c in use}))
        for b in rdr:
            t=b.filter(T(b.column('search_conducted')))
            if t.num_rows==0: continue
            agg['n']+=t.num_rows
            cf=T(t.column('contraband_found')); dr=T(t.column('contraband_drugs')); wp=T(t.column('contraband_weapons'))
            dw=pc.or_(dr,wp)
            agg['found']+=pc.sum(pc.cast(cf,pa.int64())).as_py() or 0
            agg['drugs']+=pc.sum(pc.cast(dr,pa.int64())).as_py() or 0
            agg['weap']+=pc.sum(pc.cast(wp,pa.int64())).as_py() or 0
            agg['F_but_drugsweap']+=pc.sum(pc.cast(pc.and_(pc.invert(cf),dw),pa.int64())).as_py() or 0
            agg['T_no_drugsweap']+=pc.sum(pc.cast(pc.and_(cf,pc.invert(dw)),pa.int64())).as_py() or 0
            yr=pc.utf8_slice_codeunits(t.column('date'),0,4)
            for x in pc.value_counts(yr).to_pylist():
                k=x['values'] or 'NA'; d=agg['year'].setdefault(k,[0,0]); d[0]+=x['counts']
            for x in pc.value_counts(pc.filter(yr,cf)).to_pylist():
                k=x['values'] or 'NA'; d=agg['year'].setdefault(k,[0,0]); d[1]+=x['counts']
            if 'officer_id_hash' in use:
                for x in pc.value_counts(t.column('officer_id_hash')).to_pylist():
                    k=x['values'] or 'NA'; d=agg['off'].setdefault(k,[0,0]); d[0]+=x['counts']
                for x in pc.value_counts(pc.filter(t.column('officer_id_hash'),cf)).to_pylist():
                    k=x['values'] or 'NA'; d=agg['off'].setdefault(k,[0,0]); d[1]+=x['counts']
    N=agg['n']
    print(f"\n=== {name}: {N:,} searches")
    print(f"  contraband_found TRUE: {agg['found']:,} ({100*agg['found']/N:.1f}%)   drugs {agg['drugs']:,}  weapons {agg['weap']:,}")
    print(f"  INCONSISTENT  found=FALSE but drugs/weapons TRUE : {agg['F_but_drugsweap']:,} ({100*agg['F_but_drugsweap']/N:.2f}%)")
    print(f"  found=TRUE but neither drugs nor weapons         : {agg['T_no_drugsweap']:,} ({100*agg['T_no_drugsweap']/N:.2f}%)")
    print("  hit rate by year:", ' '.join(f"{y}:{100*v[1]/max(v[0],1):.0f}%({v[0]//1000}k)" for y,v in sorted(agg['year'].items())))
    offs=[(k,v[0],v[1]) for k,v in agg['off'].items() if v[0]>=50]
    zero=[o for o in offs if o[2]==0]
    print(f"  officers with >=50 searches: {len(offs):,}; of those with ZERO recorded hits: {len(zero):,} ({100*len(zero)/max(len(offs),1):.1f}%), covering {sum(o[1] for o in zero):,} searches")
