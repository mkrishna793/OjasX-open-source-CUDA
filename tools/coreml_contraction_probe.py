"""Core ML lowering and CPU qualification of explicit joint linear-chain VJPs.

Uses dynamic weights as inputs. Does not enable on-device model training or
claim Neural Engine execution; the earlier hardware probe could not confirm it.
"""
import json,time,traceback
from pathlib import Path
import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
ROOT=Path(__file__).resolve().parents[1]


def main():
    m,k,n,p=64,128,256,8
    report={'coremltools':ct.__version__,'shape_m_k_n_p':[m,k,n,p],'configuration':'CPU_ONLY',
            'scope':'Explicit forward and all three VJPs with dynamic weights; not a Core ML training backend or ANE/GPU benchmark','results':[]}
    rng=np.random.default_rng(779)
    data={'x':(rng.normal(size=(m,k))*.1).astype(np.float16),
          'w1':(rng.normal(size=(n,k))*.03).astype(np.float16),
          'w2':(rng.normal(size=(p,n))*.03).astype(np.float16),
          'g':(rng.normal(size=(m,p))*.1).astype(np.float16)}
    x,w1,w2,g=[data[key].astype(np.float64) for key in ['x','w1','w2','g']]
    h=x@w1.T;g1=g@w2
    reference={'output':h@w2.T,'d_input':g1@w1,'d_w1':g1.T@x,'d_w2':g.T@h}
    for mode in ['native','joint']:
        item={'schedule':mode,'neural_engine_execution_confirmed':False}
        try:
            @mb.program(input_specs=[mb.TensorSpec(shape=(m,k),dtype=types.fp16),
                                    mb.TensorSpec(shape=(n,k),dtype=types.fp16),
                                    mb.TensorSpec(shape=(p,n),dtype=types.fp16),
                                    mb.TensorSpec(shape=(m,p),dtype=types.fp16)],opset_version=ct.target.macOS15)
            def program(x,w1,w2,g):
                if mode=='native':
                    h=mb.matmul(x=x,y=w1,transpose_y=True)
                    y=mb.matmul(x=h,y=w2,transpose_y=True,name='output')
                    g1=mb.matmul(x=g,y=w2)
                    dx=mb.matmul(x=g1,y=w1,name='d_input')
                    dw1=mb.matmul(x=g1,y=x,transpose_x=True,name='d_w1')
                    dw2=mb.matmul(x=g,y=h,transpose_x=True,name='d_w2')
                else:
                    effective=mb.matmul(x=w2,y=w1)
                    y=mb.matmul(x=x,y=effective,transpose_y=True,name='output')
                    t=mb.matmul(x=g,y=x,transpose_x=True)
                    dx=mb.matmul(x=g,y=effective,name='d_input')
                    dw1=mb.matmul(x=w2,y=t,transpose_x=True,name='d_w1')
                    dw2=mb.matmul(x=t,y=w1,transpose_y=True,name='d_w2')
                return y,dx,dw1,dw2
            model=ct.convert(program,convert_to='mlprogram',minimum_deployment_target=ct.target.macOS15,
                             compute_precision=ct.precision.FLOAT16,skip_model_load=True)
            path=ROOT/f'.tools/coreml-models/linear_chain_{mode}.mlpackage';path.parent.mkdir(parents=True,exist_ok=True)
            model.save(str(path));item['conversion']='passed'
            item['mil_matmuls']=sum(op.op_type=='matmul' for op in program.functions['main'].operations)
            compiled=ct.models.MLModel(str(path),compute_units=ct.ComputeUnit.CPU_ONLY)
            result=compiled.predict(data)
            item['errors']={key:{'relative_l2':float(np.linalg.norm(result[key].astype(np.float64)-ref)/np.linalg.norm(ref)),
                                 'max_abs':float(np.max(np.abs(result[key].astype(np.float64)-ref))),
                                 'within_fp16_tolerance':bool(np.allclose(result[key],ref,rtol=.01,atol=.002))}
                            for key,ref in reference.items()}
            item['status']='executed'
            durations=[]
            for _ in range(7):
                start=time.perf_counter();compiled.predict(data);durations.append(time.perf_counter()-start)
            item['median_predict_seconds']=float(np.median(durations))
            try:
                plan=ct.models.compute_plan.MLComputePlan.load_from_path(compiled.get_compiled_model_path(),compute_units=ct.ComputeUnit.CPU_ONLY)
                item['compute_plan']=[{'operation':op.operator_name,'preferred_device':type(plan.get_compute_device_usage_for_mlprogram_operation(op).preferred_compute_device).__name__}
                                      for fn in plan.model_structure.program.functions.values() for op in fn.block.operations
                                      if plan.get_compute_device_usage_for_mlprogram_operation(op) is not None]
            except Exception as error:item['compute_plan_error']=str(error)
        except Exception as error:
            item['status']='blocked';item['error']=str(error);item['traceback']=traceback.format_exc()
        report['results'].append(item)
    report['all_executed_samples_passed']=all(r.get('status')=='executed' and all(e['within_fp16_tolerance'] for e in r['errors'].values()) for r in report['results'])
    (ROOT/'reports/coreml-contraction.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
