"""Distinguish physical Apple GPU hardware from process-visible acceleration."""
import json
from pathlib import Path
import subprocess
import torch

ROOT=Path(__file__).resolve().parents[1]


def main():
    report={'pytorch':{'version':torch.__version__,'mps_built':torch.backends.mps.is_built(),
                        'mps_available':torch.backends.mps.is_available()},
            'context':'Current process; advertised devices do not certify execution',
            'neural_engine_execution_confirmed':False}
    try:
        physical=json.loads(subprocess.check_output(['system_profiler','-json','SPHardwareDataType','SPDisplaysDataType'],text=True))
        hw=physical['SPHardwareDataType'][0]
        report['physical_hardware']={k:hw.get(k) for k in ('machine_name','chip_type','physical_memory')}
        report['physical_displays']=[{k:device.get(k) for k in ('sppci_model','sppci_cores','spdisplays_metal')} for device in physical.get('SPDisplaysDataType',[])]
        # Deliberately omit serial numbers, hardware UUIDs and provisioning IDs.
    except Exception as e:report['physical_probe_error']=str(e)
    try:
        sdk=Path('/Library/Developer/CommandLineTools/SDKs/MacOSX15.4.sdk')
        if not sdk.exists():sdk=Path(subprocess.check_output(['xcrun','--show-sdk-path'],text=True).strip())
        binary=ROOT/'.tools/native-hardware-probe'
        subprocess.run(['clang','-isysroot',str(sdk),'-fobjc-arc','-framework','Foundation','-framework','Metal','-framework','CoreML',str(ROOT/'tools/hardware_probe.m'),'-o',str(binary)],check=True,capture_output=True,text=True)
        report.update(json.loads(subprocess.check_output([str(binary)],text=True)))
        report['native_probe_sdk']=str(sdk)
    except Exception as e:report['native_probe_error']=str(e)
    (ROOT/'reports/hardware.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
