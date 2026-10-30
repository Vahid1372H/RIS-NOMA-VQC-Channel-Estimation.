"""Leakage-safe chronological training and evaluation (Algorithm 1)."""
import copy, time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .qnn_models import CNNLSTM, HybridVQC, ClassicalAblation, LinearBottleneck
from .stats_analysis import regression_metrics


def chronological_indices(T,cfg):
    """Chronological 70/15/15 segments separated by W unused raw blocks."""
    a=int(T*cfg.train_fraction); b=int(T*(cfg.train_fraction+cfg.validation_fraction))
    gap=cfg.split_gap if cfg.split_gap is not None else cfg.W
    train=np.arange(0,a)
    val=np.arange(min(a+gap,T),b)
    test=np.arange(min(b+gap,T),T)
    return train,val,test


def _windows(ls,h,indices,W):
    # A target is admitted only when its whole causal input window remains inside
    # the same split: no window crosses a temporal boundary or the W-block gap.
    idx=np.asarray(indices); keep=idx[idx>=W-1]
    keep=keep[np.isin(keep-W+1,idx)]
    keep=np.array([t for t in keep if np.all(np.isin(np.arange(t-W+1,t+1),idx))],dtype=int)
    x=np.stack([ls[t-W+1:t+1] for t in keep])
    y=h[keep]
    x=np.concatenate([x.real,x.imag],axis=-1)
    y=np.concatenate([y.real,y.imag],axis=-1)
    return x.astype(np.float32),y.astype(np.float32),keep


def _stats(x,y):
    # Channel-wise statistics only from training windows; constant axes are safe.
    xm=x.mean(axis=(0,1),keepdims=True); xs=x.std(axis=(0,1),keepdims=True)
    ym=y.mean(axis=0,keepdims=True); ys=y.std(axis=0,keepdims=True)
    xs=np.where(xs<1e-8,1.0,xs); ys=np.where(ys<1e-8,1.0,ys)
    return xm,xs,ym,ys


def _build_model(name,cfg):
    if name=='cnn_lstm': return CNNLSTM(cfg.M,cfg.W)
    if name=='vqc': return HybridVQC(cfg.M,cfg.W,cfg.n_qubits,cfg.circuit_depth)
    if name=='classical_ablation': return ClassicalAblation(cfg.M,cfg.W,cfg.n_qubits)
    if name=='linear_bottleneck': return LinearBottleneck(cfg.M,cfg.W,cfg.n_qubits)
    raise ValueError(f"Unknown model: {name}")


def fit_and_evaluate(trace,cfg,model_name='cnn_lstm',seed=None):
    """Train one learned estimator; returns test metrics, history, and model."""
    seed=int(trace.seed if seed is None else seed)
    np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(cfg.torch_num_threads)
    tr,va,te=chronological_indices(len(trace.h),cfg)
    xtr,ytr,itr=_windows(trace.ls,trace.h,tr,cfg.W)
    xva,yva,iva=_windows(trace.ls,trace.h,va,cfg.W)
    xte,yte,ite=_windows(trace.ls,trace.h,te,cfg.W)
    if min(len(xtr),len(xva),len(xte))==0: raise ValueError("Not enough blocks after split gaps for W windows.")
    xm,xs,ym,ys=_stats(xtr,ytr)
    norm=lambda x:(x-xm)/xs
    target=lambda y:(y-ym)/ys
    device=torch.device(cfg.device)
    train_ds=TensorDataset(torch.from_numpy(norm(xtr)),torch.from_numpy(target(ytr)))
    loader=DataLoader(train_ds,batch_size=cfg.batch_size,shuffle=True,drop_last=False)
    xv=torch.from_numpy(norm(xva)).to(device); yv=torch.from_numpy(target(yva)).to(device)
    model=_build_model(model_name,cfg).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=cfg.learning_rate)
    loss_fn=nn.MSELoss(); best=float('inf'); best_state=None; stale=0; history=[]
    start=time.perf_counter()
    for epoch in range(cfg.max_epochs):
        model.train(); train_loss=0.0; n=0
        for xb,yb in loader:
            xb=xb.to(device); yb=yb.to(device)
            opt.zero_grad(set_to_none=True); pred=model(xb); loss=loss_fn(pred,yb)
            loss.backward(); opt.step(); train_loss+=loss.item()*len(xb); n+=len(xb)
        model.eval()
        with torch.no_grad(): val_loss=loss_fn(model(xv),yv).item()
        history.append({'epoch':epoch+1,'train_loss':train_loss/max(n,1),'validation_loss':val_loss})
        if val_loss < best-1e-10:
            best=val_loss; best_state=copy.deepcopy(model.state_dict()); stale=0
        else:
            stale+=1
            if stale>=cfg.patience: break
    elapsed=time.perf_counter()-start
    if best_state is not None: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred=model(torch.from_numpy(norm(xte)).to(device)).cpu().numpy()*ys+ym
    # Network output is [Re(h_1..M), Im(h_1..M)].
    m=cfg.M; pred_c=pred[:,:m]+1j*pred[:,m:]
    truth=trace.h[ite]
    metrics=regression_metrics(truth,pred_c)
    metrics.update({'epochs':len(history),'best_validation_loss':best,'training_seconds':elapsed,
                    'test_samples':len(ite),'seed':seed,'model':model_name,
                    'test_indices':ite,'prediction':pred_c,'truth':truth})
    return metrics,history,model


def evaluate_classical_baselines(trace,cfg):
    """LS, training-moment LMMSE, and causal Kalman test-set metrics."""
    tr,_,te=chronological_indices(len(trace.h),cfg)
    # Fit complex moments on training segment only.
    mu=trace.h[tr].mean(axis=0); var=np.mean(np.abs(trace.h[tr]-mu)**2,axis=0)
    obs=trace.ls[te]; truth=trace.h[te]
    from .channel_model import estimate_lmmse,estimate_kalman
    ls=obs
    lmmse=estimate_lmmse(obs,mu,var,trace.noise_variance)
    # Initialize the recursion from the first training observation; only past data.
    joined=np.concatenate([trace.ls[tr[-1]:tr[-1]+1],obs],axis=0)
    k=estimate_kalman(joined,mu,var,trace.noise_variance,trace.rho)[1:]
    return {name:regression_metrics(truth,val) for name,val in [('ls',ls),('lmmse',lmmse),('kalman',k)]}
