"""RIS cascaded channel, pilots, NOMA symbols, and classical estimators.

The simulator is synthetic: it follows the paper's stated Rician / Gauss-Markov
assumptions and is not a redistributable measurement campaign.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import j0

@dataclass
class ChannelTrace:
    h: np.ndarray          # [T, M], complex effective channels
    ls: np.ndarray         # [T, M], pilot LS estimates
    noise_variance: float  # complex noise variance per LS observation
    rho: float
    seed: int


def _cn(rng, shape):
    return (rng.standard_normal(shape) + 1j*rng.standard_normal(shape))/np.sqrt(2.0)


def generate_cascaded_channels(cfg, rng):
    """Generate h_j(t)=sqrt(PL_j) sum_l H_l Phi_l g_{j,l}(t).

    BS-RIS and RIS phases are fixed; each user has a geometry-fixed LoS component
    plus temporally correlated diffuse Rician fading. Large-scale gain is
    normalized so E|h_1(0)|^2 is approximately one for L=64, as in Section II-C.
    """
    M,L,T=cfg.M,cfg.L,cfg.T
    bs_ris=_cn(rng,(L,))
    ris_phase=np.exp(1j*rng.uniform(0,2*np.pi,L))
    fixed=bs_ris*ris_phase
    # A simple ULA steering vector: user-specific angle is fixed over trajectory.
    h=np.empty((T,M),dtype=np.complex128)
    common_norm=None
    rho=float(cfg.rho if cfg.rho is not None else j0(2*np.pi*cfg.f_D))
    rho=np.clip(rho,-0.999999,0.999999)
    for user in range(M):
        angle=-0.65+1.3*(user/max(M-1,1))
        steering=np.exp(1j*np.pi*np.arange(L)*np.sin(angle))
        los=steering/np.sqrt(L)
        diffuse=np.empty((T,L),dtype=np.complex128)
        diffuse[0]=_cn(rng,(L,))
        for t in range(1,T):
            diffuse[t]=rho*diffuse[t-1]+np.sqrt(1-rho*rho)*_cn(rng,(L,))
        g=(np.sqrt(cfg.rician_K/(cfg.rician_K+1))*los[None,:]
           +np.sqrt(1/(cfg.rician_K+1))*diffuse)
        cascaded=np.sum(fixed[None,:]*g,axis=1)
        d0=cfg.user_distances_m[user]
        # Relative path loss preserves unit average power for UE1 at t=0.
        distances=d0+cfg.displacement_m_per_block*np.arange(T)
        pl=(cfg.d_sr_m/cfg.d0_m)**(-cfg.pathloss_bs_ris)*(distances/cfg.d0_m)**(-cfg.pathloss_ris_user)
        pl_ref=(cfg.d_sr_m/cfg.d0_m)**(-cfg.pathloss_bs_ris)*(cfg.user_distances_m[0]/cfg.d0_m)**(-cfg.pathloss_ris_user)
        scale=np.sqrt(pl/pl_ref)
        if user==0:
            # One common channel normalization keeps user path-loss ratios intact.
            common_norm=np.sqrt(np.mean(np.abs(cascaded[:min(T,256)])**2))
        cascaded=cascaded/max(float(common_norm),1e-12)
        h[:,user]=scale*cascaded
    return h,rho


def qpsk_symbols(rng, shape):
    """Unit-energy Gray QPSK symbols; independent data for each NOMA user."""
    b=rng.integers(0,2,size=shape+(2,))
    return ((1-2*b[...,0])+1j*(1-2*b[...,1]))/np.sqrt(2.0)


def generate_noma_observations(h, cfg, snr_db, rng):
    """Generate superposition-coded downlink samples for optional BER studies."""
    h=np.asarray(h); T,M=h.shape
    a=np.asarray(cfg.power_shares[:M],dtype=float)
    s=qpsk_symbols(rng,(T,cfg.symbols_per_block,M))
    x=np.sum(np.sqrt(a)[None,None,:]*s,axis=-1)
    noise_var=10.0**(-float(snr_db)/10.0)
    n=_cn(rng,x.shape)*np.sqrt(noise_var)
    # Received waveform per user: y[t,j,n] = h[t,j] x[t,n] + noise.
    y=h[:,:,None]*x[:,None,:]+_cn(rng,(T,M,cfg.symbols_per_block))*np.sqrt(noise_var)
    return x,s,y


def pilot_transmission(h,cfg,snr_db,rng):
    """Unit-modulus pilot observation and equivalent LS statistic per block.

    The sufficient statistic is r=h+CN(0,N0/Np). SNR uses the paper's unit
    average initial near-user channel normalization and unit transmit power.
    """
    variance=10.0**(-float(snr_db)/10.0)/cfg.N_p
    ls=h+_cn(rng,h.shape)*np.sqrt(variance)
    return ls,variance


def simulate_trace(cfg,seed,snr_db,rho=None):
    """Deterministic complete channel-and-pilot trajectory for one seed/SNR."""
    rng=np.random.default_rng(int(seed))
    if rho is None:
        h,r=generate_cascaded_channels(cfg,rng)
    else:
        old=cfg.rho
        cfg.rho=float(rho)
        try: h,r=generate_cascaded_channels(cfg,rng)
        finally: cfg.rho=old
    ls,var=pilot_transmission(h,cfg,snr_db,rng)
    return ChannelTrace(h,ls,var,r,int(seed))


def estimate_lmmse(observations,mu,var_h,noise_var):
    """Per-user scalar LMMSE estimate, moments supplied from training only."""
    mu=np.asarray(mu); var_h=np.asarray(var_h)
    gain=var_h/(var_h+noise_var+1e-15)
    return mu+(observations-mu)*gain


def estimate_kalman(observations,mu,var_h,noise_var,rho,initial_variance=None):
    """Complex scalar AR(1) Kalman tracker with steady-state process variance."""
    z=np.asarray(observations); T,M=z.shape
    mu=np.asarray(mu); var_h=np.asarray(var_h)
    x=np.empty_like(z,dtype=np.complex128); p=np.broadcast_to(var_h,(M,)).astype(float).copy()
    if initial_variance is not None: p[:]=np.asarray(initial_variance)
    for t in range(T):
        if t:
            pred=mu+rho*(x[t-1]-mu)
            p=rho*rho*p+(1-rho*rho)*var_h
        else: pred=mu.copy()
        k=p/(p+noise_var+1e-15)
        x[t]=pred+k*(z[t]-pred)
        p=(1-k)*p
    return x


def estimate_windowed_lmmse(train_obs,train_h,test_windows,ridge=1e-6):
    """Joint-user causal affine W-window LMMSE via ridge least squares."""
    X=np.asarray(train_obs); Y=np.asarray(train_h); Q=np.asarray(test_windows)
    m=X.shape[1]; w=Q.shape[1]
    def feats(a):
        a=np.asarray(a).reshape(len(a),-1)
        return np.concatenate([a.real,a.imag,np.ones((len(a),1))],axis=1)
    # Fit real-valued representation for complex multivariate regression.
    A=feats(X); B=np.concatenate([Y.real,Y.imag],axis=1)
    coef=np.linalg.solve(A.T@A+ridge*np.eye(A.shape[1]),A.T@B)
    pred=feats(Q)@coef
    return pred[:,:m]+1j*pred[:,m:]
