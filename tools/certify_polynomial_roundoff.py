"""Offline pointwise scalar-roundoff certificates, not GPU GEMM certificates.

Each finite RN-like scalar operation is checked against u*abs(exact operation)
+ eta using rational arithmetic. Bounds propagate by the rules in AbstractEngine.
This simulator has substantial oracle overhead and is not a proposed runtime.
"""
from fractions import Fraction as Q
import json
import math
from pathlib import Path
import random
import struct
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(ROOT))
from test_feature_core import mm,transpose,lift,pull
from ojasx2.feature_core import feature_basis

class Trace:
    bits=32;operations=0;max_error_ratio=0.
    def __init__(self,exact,value=None,bound=None):
        self.exact=Q(exact);self.value=self.round(self.exact) if value is None else value
        self.bound=abs(self.value-self.exact) if bound is None else bound
        if abs(self.value-self.exact)>self.bound:raise AssertionError('Global propagated bound failed')
    @classmethod
    def round(cls,value):
        f=float(value)
        if cls.bits==32:
            try:f=struct.unpack('f',struct.pack('f',f))[0]
            except OverflowError:raise ValueError('Overflow rejects numerical certificate')
        if not math.isfinite(f):raise ValueError('Nonfinite result rejects numerical certificate')
        return Q(f)
    @classmethod
    def local(cls,value):
        rounded=cls.round(value)
        u=Q(1,2**(24 if cls.bits==32 else 53))
        eta=Q(1,2**(150 if cls.bits==32 else 1075))
        delta=u*abs(value)+eta
        actual=abs(rounded-value)
        if actual>delta:raise AssertionError('Scalar rounding assumption failed')
        cls.operations+=1;cls.max_error_ratio=max(cls.max_error_ratio,float(actual/delta))
        return rounded,delta
    def __add__(self,other):
        if not isinstance(other,Trace):other=Trace(other)
        v,d=self.local(self.value+other.value)
        return Trace(self.exact+other.exact,v,self.bound+other.bound+d)
    __radd__=__add__
    def __mul__(self,other):
        if not isinstance(other,Trace):other=Trace(other)
        v,d=self.local(self.value*other.value)
        return Trace(self.exact*other.exact,v,d+abs(self.value)*other.bound+abs(other.exact)*self.bound)
    __rmul__=__mul__
    def __pow__(self,degree):
        result=Trace(1)
        for _ in range(degree):result=result*self
        return result

def flatten(value):
    if isinstance(value,Trace):return [value]
    return [t for item in value for t in flatten(item)]

def main():
    rng=random.Random(819);cases=[]
    for bits in (32,64):
        Trace.bits=bits
        for degree in (2,3):
            for index in range(6):
                Trace.operations=0;Trace.max_error_ratio=0.
                def matrix(rows,cols):return [[Trace(Q(rng.randrange(-7,8),7 if index%2 else 8)) for _ in range(cols)] for _ in range(rows)]
                x=matrix(2,2);a=matrix(2,3);b=matrix(3,2);g=matrix(2,2)
                if index==0:x=[[Trace(0),Trace(0)] for _ in range(2)]
                z=mm(x,a);v=[[t**degree for t in row] for row in z]
                dz=[[degree*t**(degree-1)*u for t,u in zip(values,up)] for values,up in zip(z,mm(g,transpose(b)))]
                native=(mm(v,b),mm(dz,transpose(a)),mm(transpose(x),dz),mm(transpose(v),g))
                terms,weights=feature_basis(2,degree);t=lift(x,terms)
                h=transpose([[u*w for u,w in zip(row,weights)] for row in lift(transpose(a),terms)])
                c=mm(h,b);dc=mm(transpose(t),g);dt=mm(g,transpose(c));dh=mm(dc,transpose(b))
                da_up=[[u*w for u,w in zip(row,weights)] for row in transpose(dh)]
                core=(mm(t,c),pull(x,dt,terms),transpose(pull(transpose(a),da_up,terms)),mm(transpose(h),dc))
                na,ca=flatten(native),flatten(core)
                for left,right in zip(na,ca):
                    assert left.exact==right.exact
                    assert abs(left.value-right.value)<=left.bound+right.bound
                cases.append({'bits':bits,'degree':degree,'case':index,'checked_scalar_operations':Trace.operations,
                    'max_local_error_fraction_of_budget':Trace.max_error_ratio,
                    'max_absolute_error':max(float(abs(v.value-v.exact)) for v in na+ca),
                    'max_propagated_absolute_bound':max(float(v.bound) for v in na+ca)})
    Trace.bits=32
    try:Trace(Q(10)**20)*Trace(Q(10)**20)
    except ValueError:overflow_rejected=True
    else:raise AssertionError('Overflow certificate incorrectly accepted')
    result={'status':'passed','cases':cases,'overflow_rejected':overflow_rejected,
        'scope':'24 finite scalar traces of forward and all operand gradients; exact rational local checks plus conditional propagated error rules. No GPU/PyTorch GEMM trace refinement, uniform domain bound, or fast runtime certifier.'}
    (ROOT/'reports/polynomial-roundoff.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'passed','cases':len(cases),'scalar_operations':sum(c['checked_scalar_operations'] for c in cases),'overflow_rejected':True},indent=2))

if __name__=='__main__':main()
