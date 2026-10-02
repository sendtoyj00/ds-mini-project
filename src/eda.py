"""Descriptive battery EDA. Full-life curves stay outside the modeling pipeline."""
import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from src.preprocess import FILES, load_data, load_targets
from src.features import prepare_features, align_curve, build_feature_sets
from src.reporting import plt

COLORS = ['#2563eb', '#ea580c', '#16a34a']


def full_capacity_curves(directory, ids):
    curves = {}
    for batch, filename in FILES.items():
        with h5py.File(Path(directory) / filename, 'r') as f:
            for cid in ids:
                if not cid.startswith(f'b{batch}c'):
                    continue
                i = int(cid.split('c')[1]); s = f[f['batch']['summary'][i, 0]]
                cycle = np.asarray(s['cycle']).ravel(); q = np.asarray(s['QDischarge']).ravel()
                good = np.isfinite(cycle) & np.isfinite(q) & (q > 0)
                curves[cid] = (cycle[good], q[good])
    return curves


def knee_estimate(cycle, q):
    """Exploratory continuous two-slope fit; not a validated physical knee detector."""
    if len(cycle) < 100:
        return np.nan, np.nan
    # Subsample uniformly for bounded compute, and smooth isolated measurement spikes.
    q = pd.Series(q).rolling(11, center=True, min_periods=1).median().to_numpy()
    pos = np.linspace(0, len(cycle)-1, min(300, len(cycle))).astype(int)
    t, values = cycle[pos], q[pos]
    single = np.c_[np.ones(len(t)), t]
    linear_sse = np.sum((values-single @ np.linalg.lstsq(single, values, rcond=None)[0])**2)
    fits = []
    for k in np.linspace(t.min()+.2*np.ptp(t), t.min()+.9*np.ptp(t), 60):
        a = np.c_[np.ones(len(t)), t, np.maximum(t-k, 0)]
        coef = np.linalg.lstsq(a, values, rcond=None)[0]
        if coef[2] < 0:
            fits.append((np.sum((values-a @ coef)**2), k))
    if not fits or linear_sse <= 0:
        return np.nan, np.nan
    sse, knee = min(fits)
    gain = 1-sse/linear_sse
    return (float(knee) if gain >= .5 else np.nan), float(gain)


def run_eda(root, out):
    root, out = Path(root), Path(out); out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((root/'config/experiment.json').read_text())
    cells, excluded = load_data(root/'data'); x, meta = prepare_features(cells, cfg)
    y = load_targets(root/'data', x.index.tolist())
    table = meta.join(x).join(y.rename('cycle_life'))
    table.to_csv(out/'cell_features.csv')
    reference = y.loc[meta.batch.eq(1)]
    stats = []
    for batch, g in table.groupby('batch'):
        stats.append(dict(batch=batch, n=len(g), mean=g.cycle_life.mean(), median=g.cycle_life.median(),
                          minimum=g.cycle_life.min(), maximum=g.cycle_life.max(),
                          below_500_pct=g.cycle_life.lt(500).mean()*100,
                          short_lt550_n=int(g.cycle_life.lt(550).sum()),
                          long_ge550_n=int(g.cycle_life.ge(550).sum()),
                          outside_batch1_pct=((g.cycle_life<reference.min())|(g.cycle_life>reference.max())).mean()*100))
    pd.DataFrame(stats).to_csv(out/'lifetime_summary.csv', index=False)
    corr=[]
    for b in [0,1,2,3]:
        g=table if b==0 else table.loc[table.batch.eq(b)]
        for feature in x.columns:
            pair=g[[feature,'cycle_life']].dropna()
            corr.append(dict(batch='All' if b==0 else str(b), feature=feature, n=len(pair),
                             spearman=pair[feature].corr(pair.cycle_life, method='spearman'),
                             pearson=pair[feature].corr(pair.cycle_life),
                             pearson_log_target=pair[feature].corr(np.log10(pair.cycle_life))))
    pd.DataFrame(corr).to_csv(out/'correlations.csv',index=False)
    policy=table.groupby(['batch','policy']).cycle_life.agg(['count','mean','median','min','max']).reset_index()
    policy.to_csv(out/'policy_lifetime.csv',index=False)
    fig, axes=plt.subplots(1,3,figsize=(13,3.7),sharex=True,sharey=True)
    bins=np.linspace(300,2000,19)
    for b,ax,c in zip([1,2,3],axes,COLORS):
        vals=table.loc[table.batch.eq(b),'cycle_life'];ax.hist(vals,bins=bins,color=c,alpha=.75)
        ax.axvline(vals.median(),color='black',ls='--');ax.set(title=f'Batch {b} (n={len(vals)})',xlabel='Cycle life')
    axes[0].set_ylabel('Cells');fig.tight_layout();fig.savefig(out/'lifetime_distribution.png',dpi=150);plt.close(fig)
    curves=full_capacity_curves(root/'data',table.index)
    knee_rows=[]; plot_outliers=[]
    fig, axes=plt.subplots(2,3,figsize=(14,8))
    norm=plt.Normalize(y.min(),y.max());cmap=plt.get_cmap('viridis')
    for cell in cells:
        cid=cell.cell_id; cycle,q=curves[cid]; initial=x.loc[cid,'qd_initial_median']
        color=cmap(norm(y.loc[cid]));b=cell.batch-1
        relative=q/initial
        for cy,val in zip(cycle[(relative<.7)|(relative>1.1)],relative[(relative<.7)|(relative>1.1)]):
            plot_outliers.append(dict(cell_id=cid,batch=cell.batch,cycle=cy,relative_capacity=val))
        axes[0,b].plot(cycle,relative,color=color,alpha=.5,lw=.7)
        early=(cycle<=100);axes[1,b].plot(cycle[early],q[early]/initial,color=color,alpha=.5,lw=.7)
        knee,gain=knee_estimate(cycle,q)
        knee_rows.append(dict(cell_id=cid,batch=cell.batch,knee_cycle=knee,knee_life_ratio=knee/y.loc[cid],fit_gain=gain))
    for b in range(3):
        axes[0,b].axvline(100,color='black',ls='--');axes[0,b].set(title=f'Batch {b+1}: full life (EDA only)',xlabel='Cycle',ylabel='Relative capacity',ylim=(.7,1.1))
        outside=sum(r['batch']==b+1 for r in plot_outliers)
        axes[0,b].text(.03,.04,f'Outside view: {outside} observations',transform=axes[0,b].transAxes,fontsize=8)
        axes[1,b].set(title=f'Batch {b+1}: first 100 cycles',xlabel='Cycle',ylabel='Relative capacity',ylim=(.85,1.06))
    fig.subplots_adjust(right=.90,wspace=.32,hspace=.35)
    cax=fig.add_axes([.93,.15,.015,.70])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,label='Cycle life')
    fig.savefig(out/'capacity_degradation.png',dpi=150);plt.close(fig)
    pd.DataFrame(plot_outliers,columns=['cell_id','batch','cycle','relative_capacity']).to_csv(out/'capacity_plot_outliers.csv',index=False)
    pd.DataFrame(knee_rows).to_csv(out/'knee_diagnostics.csv',index=False)
    grid=np.linspace(*cfg['voltage_grid'])
    fig,axes=plt.subplots(1,3,figsize=(14,4),sharey=True)
    for b,ax in zip([1,2,3],axes):
        group=table.loc[table.batch.eq(b)].sort_values('cycle_life');n=len(group)//3
        for ids,label,ls in [(group.index[:n],'Short-life third','-'),(group.index[-n:],'Long-life third','--')]:
            lines=[]
            for cell in cells:
                if cell.cell_id in ids:
                    q10,_=align_curve(cell.curves.get(10),cfg);q100,_=align_curve(cell.curves.get(100),cfg)
                    if q10 is not None and q100 is not None:lines.append(q100-q10)
            a=np.array(lines);line=ax.plot(grid,np.median(a,axis=0),ls=ls,label=label)[0]
            ax.fill_between(grid,np.quantile(a,.25,axis=0),np.quantile(a,.75,axis=0),color=line.get_color(),alpha=.15)
        ax.set(title=f'Batch {b}',xlabel='Voltage (V)');ax.legend(fontsize=8)
    axes[0].set_ylabel('Q100 - Q10 (Ah)');fig.tight_layout();fig.savefig(out/'delta_q_curves.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for b,c in zip([1,2,3],COLORS):
        g=table.loc[table.batch.eq(b)]
        axes[0].scatter(g.log10_dq_var,g.cycle_life,label=f'Batch {b}',color=c)
        axes[1].scatter(g.log10_dq_var,np.log10(g.cycle_life),color=c)
        axes[2].scatter(g.peak_c_rate,g.cycle_life,color=c)
    for ax,xf,yf in zip(axes,['log10_dq_var','log10_dq_var','Peak C-rate'],['Cycle life','log10(cycle life)','Cycle life']):ax.set(xlabel=xf,ylabel=yf)
    axes[0].legend();fig.tight_layout();fig.savefig(out/'feature_lifetime.png',dpi=150);plt.close(fig)
    features=build_feature_sets()['M4'];fig,axes=plt.subplots(2,4,figsize=(14,7))
    for ax,f in zip(axes.ravel(),features):
        ax.boxplot([table.loc[table.batch.eq(b),f].dropna() for b in [1,2,3]],tick_labels=['B1','B2','B3']);ax.set_title(f,fontsize=9)
    axes.ravel()[-1].axis('off');fig.tight_layout();fig.savefig(out/'feature_distributions.png',dpi=150);plt.close(fig)
    return table

if __name__ == '__main__':
    run_eda(Path.cwd(),Path('results/eda'))
