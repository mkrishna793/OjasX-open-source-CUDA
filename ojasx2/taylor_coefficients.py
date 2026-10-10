"""Generated coefficient maps. See discover_nonlinear_taylor.py; not GPU-qualified by generation alone."""
import torch
import math
BUILDERS = {}
PULLS = {}
TERMS = {}
COUNTS = {}
def build_activation_1(a,v,c):
    a0=a[0]
    return torch.stack([a0/2,a0**2*c])
def pull_activation_1(a,v,c,dh):
    a0=a[0]
    da=torch.stack([(1/2)*dh[0]+(2*a0*c)*dh[1]])
    dv=None
    return da,dv
BUILDERS[('activation', 1)]=build_activation_1
PULLS[('activation', 1)]=pull_activation_1
TERMS[('activation', 1)]=[(0,), (0, 0)]
COUNTS[('activation', 1)]={'coefficient_mul': 3, 'coefficient_add': 0, 'pull_mul': 4, 'pull_add': 1}
def build_activation_2(a,v,c):
    a0=a[0]
    a1=a[1]
    return torch.stack([a1/2,a0/2,a1**2*c,2*a0*a1*c,a0**2*c])
def pull_activation_2(a,v,c,dh):
    a0=a[0]
    a1=a[1]
    da=torch.stack([(1/2)*dh[1]+(2*a1*c)*dh[3]+(2*a0*c)*dh[4],(1/2)*dh[0]+(2*a1*c)*dh[2]+(2*a0*c)*dh[3]])
    dv=None
    return da,dv
BUILDERS[('activation', 2)]=build_activation_2
PULLS[('activation', 2)]=pull_activation_2
TERMS[('activation', 2)]=[(1,), (0,), (1, 1), (0, 1), (0, 0)]
COUNTS[('activation', 2)]={'coefficient_mul': 9, 'coefficient_add': 0, 'pull_mul': 14, 'pull_add': 4}
def build_activation_4(a,v,c):
    a0=a[0]
    a1=a[1]
    a2=a[2]
    a3=a[3]
    return torch.stack([a3/2,a2/2,a1/2,a0/2,a3**2*c,2*a2*a3*c,a2**2*c,2*a1*a3*c,2*a1*a2*c,a1**2*c,2*a0*a3*c,2*a0*a2*c,2*a0*a1*c,a0**2*c])
def pull_activation_4(a,v,c,dh):
    a0=a[0]
    a1=a[1]
    a2=a[2]
    a3=a[3]
    da=torch.stack([(1/2)*dh[3]+(2*a3*c)*dh[10]+(2*a2*c)*dh[11]+(2*a1*c)*dh[12]+(2*a0*c)*dh[13],(1/2)*dh[2]+(2*a3*c)*dh[7]+(2*a2*c)*dh[8]+(2*a1*c)*dh[9]+(2*a0*c)*dh[12],(1/2)*dh[1]+(2*a3*c)*dh[5]+(2*a2*c)*dh[6]+(2*a1*c)*dh[8]+(2*a0*c)*dh[11],(1/2)*dh[0]+(2*a3*c)*dh[4]+(2*a2*c)*dh[5]+(2*a1*c)*dh[7]+(2*a0*c)*dh[10]])
    dv=None
    return da,dv
BUILDERS[('activation', 4)]=build_activation_4
PULLS[('activation', 4)]=pull_activation_4
TERMS[('activation', 4)]=[(3,), (2,), (1,), (0,), (3, 3), (2, 3), (2, 2), (1, 3), (1, 2), (1, 1), (0, 3), (0, 2), (0, 1), (0, 0)]
COUNTS[('activation', 4)]={'coefficient_mul': 30, 'coefficient_add': 0, 'pull_mul': 52, 'pull_add': 16}
def build_swiglu_1(a,v,c):
    a0=a[0]
    v0=v[0]
    return torch.stack([a0*v0/2,a0**2*v0/4])
def pull_swiglu_1(a,v,c,dh):
    a0=a[0]
    v0=v[0]
    da=torch.stack([(v0/2)*dh[0]+(a0*v0/2)*dh[1]])
    dv=torch.stack([(a0/2)*dh[0]+(a0**2/4)*dh[1]])
    return da,dv
BUILDERS[('swiglu', 1)]=build_swiglu_1
PULLS[('swiglu', 1)]=pull_swiglu_1
TERMS[('swiglu', 1)]=[(0, 0), (0, 0, 0)]
COUNTS[('swiglu', 1)]={'coefficient_mul': 5, 'coefficient_add': 0, 'pull_mul': 10, 'pull_add': 2}
def build_swiglu_2(a,v,c):
    a0=a[0]
    a1=a[1]
    v0=v[0]
    v1=v[1]
    return torch.stack([a1*v1/2,a0*v1/2 + a1*v0/2,a0*v0/2,a1**2*v1/4,a0*a1*v1/2 + a1**2*v0/4,a0**2*v1/4 + a0*a1*v0/2,a0**2*v0/4])
def pull_swiglu_2(a,v,c,dh):
    a0=a[0]
    a1=a[1]
    v0=v[0]
    v1=v[1]
    da=torch.stack([(v1/2)*dh[1]+(v0/2)*dh[2]+(a1*v1/2)*dh[4]+(a0*v1/2 + a1*v0/2)*dh[5]+(a0*v0/2)*dh[6],(v1/2)*dh[0]+(v0/2)*dh[1]+(a1*v1/2)*dh[3]+(a0*v1/2 + a1*v0/2)*dh[4]+(a0*v0/2)*dh[5]])
    dv=torch.stack([(a1/2)*dh[1]+(a0/2)*dh[2]+(a1**2/4)*dh[4]+(a0*a1/2)*dh[5]+(a0**2/4)*dh[6],(a1/2)*dh[0]+(a0/2)*dh[1]+(a1**2/4)*dh[3]+(a0*a1/2)*dh[4]+(a0**2/4)*dh[5]])
    return da,dv
BUILDERS[('swiglu', 2)]=build_swiglu_2
PULLS[('swiglu', 2)]=pull_swiglu_2
TERMS[('swiglu', 2)]=[(1, 1), (0, 1), (0, 0), (1, 1, 1), (0, 1, 1), (0, 0, 1), (0, 0, 0)]
COUNTS[('swiglu', 2)]={'coefficient_mul': 26, 'coefficient_add': 3, 'pull_mul': 56, 'pull_add': 18}
def build_swiglu_4(a,v,c):
    a0=a[0]
    a1=a[1]
    a2=a[2]
    a3=a[3]
    v0=v[0]
    v1=v[1]
    v2=v[2]
    v3=v[3]
    return torch.stack([a3*v3/2,a2*v3/2 + a3*v2/2,a2*v2/2,a1*v3/2 + a3*v1/2,a1*v2/2 + a2*v1/2,a1*v1/2,a0*v3/2 + a3*v0/2,a0*v2/2 + a2*v0/2,a0*v1/2 + a1*v0/2,a0*v0/2,a3**2*v3/4,a2*a3*v3/2 + a3**2*v2/4,a2**2*v3/4 + a2*a3*v2/2,a2**2*v2/4,a1*a3*v3/2 + a3**2*v1/4,a1*a2*v3/2 + a1*a3*v2/2 + a2*a3*v1/2,a1*a2*v2/2 + a2**2*v1/4,a1**2*v3/4 + a1*a3*v1/2,a1**2*v2/4 + a1*a2*v1/2,a1**2*v1/4,a0*a3*v3/2 + a3**2*v0/4,a0*a2*v3/2 + a0*a3*v2/2 + a2*a3*v0/2,a0*a2*v2/2 + a2**2*v0/4,a0*a1*v3/2 + a0*a3*v1/2 + a1*a3*v0/2,a0*a1*v2/2 + a0*a2*v1/2 + a1*a2*v0/2,a0*a1*v1/2 + a1**2*v0/4,a0**2*v3/4 + a0*a3*v0/2,a0**2*v2/4 + a0*a2*v0/2,a0**2*v1/4 + a0*a1*v0/2,a0**2*v0/4])
def pull_swiglu_4(a,v,c,dh):
    a0=a[0]
    a1=a[1]
    a2=a[2]
    a3=a[3]
    v0=v[0]
    v1=v[1]
    v2=v[2]
    v3=v[3]
    da=torch.stack([(v3/2)*dh[6]+(v2/2)*dh[7]+(v1/2)*dh[8]+(v0/2)*dh[9]+(a3*v3/2)*dh[20]+(a2*v3/2 + a3*v2/2)*dh[21]+(a2*v2/2)*dh[22]+(a1*v3/2 + a3*v1/2)*dh[23]+(a1*v2/2 + a2*v1/2)*dh[24]+(a1*v1/2)*dh[25]+(a0*v3/2 + a3*v0/2)*dh[26]+(a0*v2/2 + a2*v0/2)*dh[27]+(a0*v1/2 + a1*v0/2)*dh[28]+(a0*v0/2)*dh[29],(v3/2)*dh[3]+(v2/2)*dh[4]+(v1/2)*dh[5]+(v0/2)*dh[8]+(a3*v3/2)*dh[14]+(a2*v3/2 + a3*v2/2)*dh[15]+(a2*v2/2)*dh[16]+(a1*v3/2 + a3*v1/2)*dh[17]+(a1*v2/2 + a2*v1/2)*dh[18]+(a1*v1/2)*dh[19]+(a0*v3/2 + a3*v0/2)*dh[23]+(a0*v2/2 + a2*v0/2)*dh[24]+(a0*v1/2 + a1*v0/2)*dh[25]+(a0*v0/2)*dh[28],(v3/2)*dh[1]+(v2/2)*dh[2]+(v1/2)*dh[4]+(v0/2)*dh[7]+(a3*v3/2)*dh[11]+(a2*v3/2 + a3*v2/2)*dh[12]+(a2*v2/2)*dh[13]+(a1*v3/2 + a3*v1/2)*dh[15]+(a1*v2/2 + a2*v1/2)*dh[16]+(a1*v1/2)*dh[18]+(a0*v3/2 + a3*v0/2)*dh[21]+(a0*v2/2 + a2*v0/2)*dh[22]+(a0*v1/2 + a1*v0/2)*dh[24]+(a0*v0/2)*dh[27],(v3/2)*dh[0]+(v2/2)*dh[1]+(v1/2)*dh[3]+(v0/2)*dh[6]+(a3*v3/2)*dh[10]+(a2*v3/2 + a3*v2/2)*dh[11]+(a2*v2/2)*dh[12]+(a1*v3/2 + a3*v1/2)*dh[14]+(a1*v2/2 + a2*v1/2)*dh[15]+(a1*v1/2)*dh[17]+(a0*v3/2 + a3*v0/2)*dh[20]+(a0*v2/2 + a2*v0/2)*dh[21]+(a0*v1/2 + a1*v0/2)*dh[23]+(a0*v0/2)*dh[26]])
    dv=torch.stack([(a3/2)*dh[6]+(a2/2)*dh[7]+(a1/2)*dh[8]+(a0/2)*dh[9]+(a3**2/4)*dh[20]+(a2*a3/2)*dh[21]+(a2**2/4)*dh[22]+(a1*a3/2)*dh[23]+(a1*a2/2)*dh[24]+(a1**2/4)*dh[25]+(a0*a3/2)*dh[26]+(a0*a2/2)*dh[27]+(a0*a1/2)*dh[28]+(a0**2/4)*dh[29],(a3/2)*dh[3]+(a2/2)*dh[4]+(a1/2)*dh[5]+(a0/2)*dh[8]+(a3**2/4)*dh[14]+(a2*a3/2)*dh[15]+(a2**2/4)*dh[16]+(a1*a3/2)*dh[17]+(a1*a2/2)*dh[18]+(a1**2/4)*dh[19]+(a0*a3/2)*dh[23]+(a0*a2/2)*dh[24]+(a0*a1/2)*dh[25]+(a0**2/4)*dh[28],(a3/2)*dh[1]+(a2/2)*dh[2]+(a1/2)*dh[4]+(a0/2)*dh[7]+(a3**2/4)*dh[11]+(a2*a3/2)*dh[12]+(a2**2/4)*dh[13]+(a1*a3/2)*dh[15]+(a1*a2/2)*dh[16]+(a1**2/4)*dh[18]+(a0*a3/2)*dh[21]+(a0*a2/2)*dh[22]+(a0*a1/2)*dh[24]+(a0**2/4)*dh[27],(a3/2)*dh[0]+(a2/2)*dh[1]+(a1/2)*dh[3]+(a0/2)*dh[6]+(a3**2/4)*dh[10]+(a2*a3/2)*dh[11]+(a2**2/4)*dh[12]+(a1*a3/2)*dh[14]+(a1*a2/2)*dh[15]+(a1**2/4)*dh[17]+(a0*a3/2)*dh[20]+(a0*a2/2)*dh[21]+(a0*a1/2)*dh[23]+(a0**2/4)*dh[26]])
    return da,dv
BUILDERS[('swiglu', 4)]=build_swiglu_4
PULLS[('swiglu', 4)]=pull_swiglu_4
TERMS[('swiglu', 4)]=[(3, 3), (2, 3), (2, 2), (1, 3), (1, 2), (1, 1), (0, 3), (0, 2), (0, 1), (0, 0), (3, 3, 3), (2, 3, 3), (2, 2, 3), (2, 2, 2), (1, 3, 3), (1, 2, 3), (1, 2, 2), (1, 1, 3), (1, 1, 2), (1, 1, 1), (0, 3, 3), (0, 2, 3), (0, 2, 2), (0, 1, 3), (0, 1, 2), (0, 1, 1), (0, 0, 3), (0, 0, 2), (0, 0, 1), (0, 0, 0)]
COUNTS[('swiglu', 4)]={'coefficient_mul': 152, 'coefficient_add': 26, 'pull_mul': 352, 'pull_add': 128}
