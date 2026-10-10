"""Bounded MPS-only checks of ordinary-model FX rewriting and conservative fallback."""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
import copy,json,sys
from pathlib import Path
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gpu_contracts import GPUOnly,require_gpu,compare,hashes
from ojasx2.loss_fx import rewrite_linear_cross_entropy


class Model(torch.nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind
        self.a=torch.nn.Linear(8,16,device='mps');self.head=torch.nn.Linear(16,31,device='mps')
    def forward(self,x,t):
        h=F.gelu(self.a(x))
        z=F.linear(h,self.head.weight,self.head.bias) if self.kind=='functional' else self.head(h)
        loss=F.cross_entropy(z,t,label_smoothing=.1,reduction='none' if self.kind=='unreduced' else 'mean')
        return loss+z.square().mean() if self.kind=='observed_logits' else loss


def main():
    require_gpu();torch.manual_seed(882);rows=[]
    with GPUOnly() as guard:
        for kind in ('module','functional','observed_logits','soft_targets','unreduced','higher_order','hook'):
            model=Model(kind);gm=torch.fx.symbolic_trace(copy.deepcopy(model))
            if kind=='hook':gm.head.register_forward_hook(lambda m,a,y:y)
            baseline=copy.deepcopy(gm)
            ledger=rewrite_linear_cross_entropy(gm,accuracy='algebraic',
                derivative_order=2 if kind=='higher_order' else 1,head_budget=31*4*6*3)
            x=torch.randn(11,8,device='mps',requires_grad=True);target=torch.arange(11,device='mps')%31;target[:5]=-100
            if kind=='soft_targets':target=torch.softmax(torch.randn(11,31,device='mps'),-1)
            outputs=[]
            for module in (baseline,gm):
                loss=module(x,target);ps=(x,)+tuple(module.parameters())
                gs=torch.autograd.grad(loss.sum(),ps,create_graph=kind=='higher_order')
                out=(loss.detach(),)+gs
                if kind=='higher_order':out+=torch.autograd.grad(sum(g.square().sum() for g in gs),ps)
                outputs.append(out)
            es=[compare(a,b) for a,b in zip(outputs[1],outputs[0])]
            expected='native' if kind in ('observed_logits','hook') else 'loss_compiler'
            passed=all(e['passed'] for e in es) and all(r['route']==expected for r in ledger) and bool(ledger)
            rows.append(dict(name=kind,ledger=ledger,errors=es,passed=passed))
    report=dict(status='passed_scoped_loss_fx' if all(r['passed'] for r in rows) else 'failed_loss_fx',
        device='mps',cpu_ml_execution=False,cpu_fallback_enabled=False,cases=rows,guarded_dispatches=guard.calls,
        scope='Seven bounded FX graphs; no arbitrary Python effects, dynamic trace capture or portable backend certification',
        source_sha256=hashes(['tools/test_loss_fx_gpu.py','tools/gpu_contracts.py','ojasx2/loss_compiler.py','ojasx2/loss_fx.py']))
    (ROOT/'reports/loss-fx-gpu.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'],[r['name'] for r in rows if not r['passed']])
    if report['status'].startswith('failed'):raise SystemExit(1)

if __name__=='__main__':main()
