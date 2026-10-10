"""Small explicit-GPU forward/VJP check; never substitutes CPU execution."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='reports/mlx-gpu-probe-current.json')
    args = parser.parse_args()
    report = {'mlx': importlib.metadata.version('mlx'),
              'requested_device': 'gpu', 'gpu_execution_confirmed': False,
              'status': 'blocked', 'stage': 'import'}
    try:
        import mlx.core as mx
        import numpy as np
        report['metal_available'] = mx.metal.is_available()
        report['stage'] = 'explicit_gpu_execution'
        mx.set_default_device(mx.gpu)
        rng = np.random.default_rng(20261007)
        arrays = [(rng.normal(size=s) * 0.1).astype(np.float32)
                  for s in [(32, 16), (16, 64), (64, 8), (32, 8)]]
        x, a, b, g = [mx.array(v) for v in arrays]
        xd, ad, bd, gd = [v.astype(np.float64) for v in arrays]
        golden = [xd @ ad @ bd, gd @ bd.T @ ad.T,
                  xd.T @ gd @ bd.T, (xd @ ad).T @ gd]
        report['cases'] = []
        for name, f in [('native', lambda x, a, b: (x @ a) @ b),
                        ('reassociated', lambda x, a, b: x @ (a @ b))]:
            outputs, gradients = mx.vjp(f, [x, a, b], [g])
            values = list(outputs) + list(gradients)
            mx.eval(*values)
            mx.synchronize()
            report['gpu_execution_confirmed'] = True
            got = [np.array(v).astype(np.float64) for v in values]
            errors = [{'max_abs': float(np.max(np.abs(v - r))),
                       'relative_l2': float(np.linalg.norm(v - r) / np.linalg.norm(r))}
                      for v, r in zip(got, golden)]
            passed = len(got) == 4 and all(
                np.allclose(v, r, rtol=1e-4, atol=1e-5)
                for v, r in zip(got, golden))
            report['cases'].append({'case': name, 'output_and_three_gradients': errors,
                                    'within_sample_tolerance': bool(passed)})
        report['status'] = ('passed' if all(c['within_sample_tolerance']
                                           for c in report['cases']) else 'failed')
        report['stage'] = 'complete'
        report['scope'] = 'One FP32 sample; no speed, memory, or universal accuracy claim.'
    except Exception as exc:
        report['error'] = str(exc)
        report['traceback'] = traceback.format_exc()
    target = Path(__file__).resolve().parents[1] / args.output
    target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
