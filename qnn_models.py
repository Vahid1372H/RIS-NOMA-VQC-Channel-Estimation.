"""CNN-LSTM, matched classical ablation, and exact four-qubit state-vector VQC.

Circuit convention: RY angle encoding, trainable RY then RZ per layer, directed
CNOT ring, and Pauli-Z expectation readout. The simulator is differentiable in
PyTorch; ``parameter_shift_jacobian`` provides the hardware-style shift rule.
"""
import math
import torch
from torch import nn

class TemporalFrontEnd(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.conv1=nn.Conv1d(features,16,3,padding=1)
        self.conv2=nn.Conv1d(16,16,3,padding=1)
        self.lstm=nn.LSTM(16,16,num_layers=2,batch_first=True)
    def forward(self,x):
        x=x.transpose(1,2)
        x=torch.relu(self.conv1(x)); x=torch.relu(self.conv2(x))
        _,(hn,_)=self.lstm(x.transpose(1,2))
        return hn[-1]

class CNNLSTM(nn.Module):
    """Paper baseline: 2 Conv1D(16,k=3), 2-layer LSTM(16), linear 2M."""
    def __init__(self,M=2,window=10):
        super().__init__(); self.M=M; self.front=TemporalFrontEnd(2*M)
        self.head=nn.Linear(16,2*M)
    def forward(self,x): return self.head(self.front(x))


def _single_qubit(state, gate, q, n):
    shape=[state.shape[0]]+[2]*n
    t=state.reshape(shape).movedim(1+q,-1)
    t=(torch.einsum('b...j,bij->b...i',t,gate) if gate.ndim==3 else torch.einsum('...j,ij->...i',t,gate))
    return t.movedim(-1,1+q).reshape(state.shape)

def _ry(theta,device,dtype):
    c=torch.cos(theta/2); s=torch.sin(theta/2)
    return torch.stack((torch.stack((c,-s),dim=-1),torch.stack((s,c),dim=-1)),dim=-2).to(device=device,dtype=dtype)
def _rz(theta,device,dtype):
    return torch.diag(torch.stack([torch.exp(-0.5j*theta),torch.exp(0.5j*theta)])).to(device=device,dtype=dtype)

def _cnot(state,control,target,n):
    ids=torch.arange(2**n,device=state.device)
    mask=1<<control; flip=1<<target
    perm=torch.where((ids&mask)!=0,ids^flip,ids)
    return state.index_select(1,perm)

class StateVectorVQC(nn.Module):
    def __init__(self,n_qubits=4,depth=3):
        super().__init__()
        self.n_qubits=n_qubits; self.depth=depth
        self.theta=nn.Parameter(0.05*torch.randn(depth,n_qubits,2))
        basis=torch.arange(2**n_qubits)
        self.register_buffer('zsign',torch.stack([1-2*((basis>>q)&1) for q in range(n_qubits)]).float())
    def forward(self,angles):
        b=angles.shape[0]; n=self.n_qubits; dev=angles.device
        state=torch.zeros((b,2**n),device=dev,dtype=torch.complex64 if angles.dtype==torch.float32 else torch.complex128)
        state[:,0]=1
        for q in range(n): state=_single_qubit(state,_ry(angles[:,q] if angles.ndim==2 else angles[q],dev,state.dtype),q,n)
        for layer in range(self.depth):
            for q in range(n):
                state=_single_qubit(state,_ry(self.theta[layer,q,0],dev,state.dtype),q,n)
                state=_single_qubit(state,_rz(self.theta[layer,q,1],dev,state.dtype),q,n)
            for q in range(n): state=_cnot(state,q,(q+1)%n,n)
        probs=state.abs().square()
        return probs@self.zsign.to(dtype=probs.dtype).T
    def parameter_shift_jacobian(self,angles,which='theta'):
        """Return d expectation / d selected scalar circuit parameter by shift rule.

        ``which`` can be 'theta' (all trainable gates) or 'angles' (encoding).
        Returns a tensor [batch, qubit, *parameter_shape]. Intended for verification,
        not as the default large-batch training path.
        """
        if which=='theta': p=self.theta; shape=tuple(p.shape)
        elif which=='angles': p=angles; shape=tuple(p.shape)
        else: raise ValueError("which must be 'theta' or 'angles'")
        grad=torch.empty((angles.shape[0],self.n_qubits)+shape,device=angles.device,dtype=angles.dtype)
        flat=p.reshape(-1)
        for i in range(flat.numel()):
            plus=p.clone().reshape(-1); minus=plus.clone(); plus[i]+=math.pi/2; minus[i]-=math.pi/2
            if which=='theta':
                old=self.theta
                self.theta=nn.Parameter(plus.reshape(shape),requires_grad=True); yp=self(angles)
                self.theta=nn.Parameter(minus.reshape(shape),requires_grad=True); ym=self(angles)
                self.theta=old
            else:
                ap=plus.reshape(shape); am=minus.reshape(shape)
                yp=self(ap); ym=self(am)
            grad.reshape(angles.shape[0],self.n_qubits,-1)[:,:,i]=(yp-ym)/2
        return grad

class HybridVQC(nn.Module):
    def __init__(self,M=2,window=10,n_qubits=4,depth=3):
        super().__init__(); self.M=M; self.front=TemporalFrontEnd(2*M)
        self.encoder=nn.Linear(16,n_qubits); self.circuit=StateVectorVQC(n_qubits,depth)
        self.readout=nn.Linear(n_qubits,2*M)
    def forward(self,x): return self.readout(self.circuit(self.encoder(self.front(x))))

class ClassicalAblation(nn.Module):
    """Matched head: circuit replaced only by tanh(Bz+b), with n_q^2+n_q params."""
    def __init__(self,M=2,window=10,n_qubits=4):
        super().__init__(); self.front=TemporalFrontEnd(2*M)
        self.encoder=nn.Linear(16,n_qubits)
        self.replacement=nn.Linear(n_qubits,n_qubits)
        self.readout=nn.Linear(n_qubits,2*M)
    def forward(self,x): return self.readout(torch.tanh(self.replacement(self.encoder(self.front(x)))))

class LinearBottleneck(CNNLSTM):
    """CNN-LSTM with width-n bottleneck, a control for bounded-head effects."""
    def __init__(self,M=2,window=10,width=4):
        super().__init__(M,window); self.bottleneck=nn.Linear(16,width); self.head=nn.Linear(width,2*M)
    def forward(self,x): return self.head(self.bottleneck(self.front(x)))

