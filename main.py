"""Small Monte Carlo demonstration; use package defaults for Table III settings."""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from .config import SimulationConfig
from .channel_model import simulate_trace
from .train_evaluate import fit_and_evaluate,evaluate_classical_baselines


def run(output_dir='results',small=True,seeds=None,snr_points=None,models=None):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    cfg=SimulationConfig()
    if small:
        cfg=cfg.with_overrides(T=600,W=10,seeds=(11,23),snr_db=(0.0,10.0),max_epochs=12,patience=4)
    seeds=tuple(seeds or cfg.seeds[:2]); snr_points=tuple(snr_points or (cfg.snr_db[:2] if small else cfg.snr_db))
    models=tuple(models or ('cnn_lstm','vqc','classical_ablation'))
    records=[]
    for snr in snr_points:
        for seed in seeds:
            trace=simulate_trace(cfg,seed,snr)
            base=evaluate_classical_baselines(trace,cfg)
            for name,metric in base.items():
                records.append({'snr_db':snr,'seed':seed,'model':name,**metric})
            for name in models:
                result,history,model=fit_and_evaluate(trace,cfg,name,seed)
                # Keep portable state dictionaries, not pickled model objects.
                checkpoint=out/f'{name}_snr{snr:g}_seed{seed}.pt'
                torch.save({'model':name,'state_dict':model.state_dict(),'snr_db':snr,'seed':seed,
                            'config':cfg.__dict__},checkpoint)
                records.append({'snr_db':snr,'seed':seed,'model':name,
                                'nmse':result['nmse'],'nmse_db':result['nmse_db'],'r2':result['r2'],
                                'epochs':result['epochs'],'training_seconds':result['training_seconds'],
                                'checkpoint':str(checkpoint)})
                print(f"SNR={snr:5.1f} dB seed={seed:3d} {name:20s} NMSE={result['nmse_db']:7.2f} dB")
    # JSON-friendly summary excludes per-sample arrays by construction.
    with (out/'metrics.json').open('w',encoding='utf-8') as f: json.dump(records,f,indent=2)
    print(f"Saved {len(records)} records to {out/'metrics.json'}")
    return records

if __name__=='__main__':
    ap=argparse.ArgumentParser(description='RIS-NOMA VQC simulation')
    ap.add_argument('--full',action='store_true',help='Use T=6000, configured SNRs and seed list.')
    ap.add_argument('--output',default='results'); args=ap.parse_args()
    run(args.output,small=not args.full)
