"""Public Core ML forward/VJP graph and planned-device attribution.

Run with the project-local Apple environment. Advertising an ANE, or allowing
it in configuration, is not evidence that it executed the matrix operations.
"""
import json
import argparse
from pathlib import Path
import time
import traceback
import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--m',type=int,default=64)
    parser.add_argument('--k',type=int,default=128)
    parser.add_argument('--n',type=int,default=128)
    parser.add_argument('--output',default='reports/coreml-probe.json')
    args=parser.parse_args()
    report={'coremltools':ct.__version__,'numpy':np.__version__,
            'scope':'Explicit forward and VJP as a Core ML graph; not an automatic Neural Engine training backend',
            'results':[]}
    try:
        m,k,n=args.m,args.k,args.n
        report['shape']=[m,k,n]
        @mb.program(input_specs=[mb.TensorSpec(shape=(m,k),dtype=types.fp16),
                                mb.TensorSpec(shape=(k,n),dtype=types.fp16),
                                mb.TensorSpec(shape=(m,n),dtype=types.fp16)],opset_version=ct.target.macOS15)
        def program(a,b,g):
            y=mb.matmul(x=a,y=b,name='forward')
            da=mb.matmul(x=g,y=b,transpose_y=True,name='d_input')
            db=mb.matmul(x=a,y=g,transpose_x=True,name='d_weight')
            return y,da,db
        model=ct.convert(program,convert_to='mlprogram',minimum_deployment_target=ct.target.macOS15,
                         compute_precision=ct.precision.FLOAT16,skip_model_load=True)
        path=ROOT/f'.tools/coreml-models/joint_vjp_{m}_{k}_{n}.mlpackage'
        path.parent.mkdir(parents=True,exist_ok=True)
        model.save(str(path))
        rng=np.random.default_rng(20261006)
        data={'a':(rng.normal(size=(m,k))*0.1).astype(np.float16),
              'b':(rng.normal(size=(k,n))*0.1).astype(np.float16),
              'g':(rng.normal(size=(m,n))*0.1).astype(np.float16)}
        golden={'forward':data['a'].astype(np.float64)@data['b'].astype(np.float64),
                'd_input':data['g'].astype(np.float64)@data['b'].astype(np.float64).T,
                'd_weight':data['a'].astype(np.float64).T@data['g'].astype(np.float64)}
        for name,units in [('cpu',ct.ComputeUnit.CPU_ONLY),('cpu_and_neural_engine',ct.ComputeUnit.CPU_AND_NE)]:
            item={'configuration':name,'neural_engine_execution_confirmed':False}
            try:
                start=time.perf_counter()
                compiled=ct.models.MLModel(str(path),compute_units=units)
                item['load_seconds']=time.perf_counter()-start
                result=compiled.predict(data)
                item['errors']={key:{'max_abs':float(np.max(np.abs(result[key].astype(np.float64)-ref))),
                                    'relative_l2':float(np.linalg.norm(result[key].astype(np.float64)-ref)/np.linalg.norm(ref))}
                                for key,ref in golden.items()}
                item['within_fp16_test_tolerance']=all(np.allclose(result[key],ref,rtol=0.01,atol=0.002) for key,ref in golden.items())
                durations=[]
                for _ in range(7):
                    begin=time.perf_counter();compiled.predict(data);durations.append(time.perf_counter()-begin)
                item['median_predict_seconds']=float(np.median(durations))
                try:
                    plan=ct.models.compute_plan.MLComputePlan.load_from_path(compiled.get_compiled_model_path(),compute_units=units)
                    operations=[]
                    for func in plan.model_structure.program.functions.values():
                        for op in func.block.operations:
                            usage=plan.get_compute_device_usage_for_mlprogram_operation(op)
                            operations.append({'operator':op.operator_name,'preferred_device':type(usage.preferred_compute_device).__name__ if usage else None,
                                               'supported_devices':[type(d).__name__ for d in usage.supported_compute_devices] if usage else []})
                    item['compute_plan']=operations
                    item['neural_engine_planned_for_matmul']=any(o['operator'].endswith('matmul') and 'NeuralEngine' in str(o['preferred_device']) for o in operations)
                    item['attribution_note']='Compute plan is anticipated placement; direct runtime activity profiling is still required for physical execution confirmation.'
                except Exception as e:
                    item['compute_plan_error']=str(e)
                item['status']='executed'
            except Exception as e:
                item['status']='blocked';item['error']=str(e)
                item['traceback']=traceback.format_exc()
            report['results'].append(item)
    except Exception as e:
        report['setup_error']=str(e);report['setup_traceback']=traceback.format_exc()
    target=ROOT/args.output
    target.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
