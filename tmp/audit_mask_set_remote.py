# -*- coding: utf-8 -*-
from pathlib import Path
import json
import cv2
import numpy as np
import torch

root = Path('/root/autodl-tmp/segmentationv2_maskset_20260910')
warm = torch.load(root / 'outputs/20260910_mask_set_stage_smoke/best_warmup.pth', map_location='cpu', weights_only=False)
joint = torch.load(root / 'outputs/20260910_mask_set_stage_smoke/best_joint.pth', map_location='cpu', weights_only=False)
ssl = torch.load(root / 'outputs/20260908_133740_ssl60/best_lora.pth', map_location='cpu', weights_only=False)
ssl = ssl.get('lora_state_dict', ssl)
warm_diffs = {k: float((v - ssl[k]).abs().max()) for k, v in warm['lora_state_dict'].items()}
joint_diffs = {k: float((v - warm['lora_state_dict'][k]).abs().max()) for k, v in joint['lora_state_dict'].items()}
assert not any(warm_diffs.values())
assert any(joint_diffs.values())
images = []
for path in (root / 'outputs/20260910_mask_set_overfit_native_export/predictions').glob('*_inst.png'):
    array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    classes = json.loads(path.with_name(path.name.replace('_inst.png', '_class.json')).read_text())
    assert array.dtype == np.uint16 and array.shape == (1936, 2584)
    assert set(map(int, classes)) == set(map(int, np.unique(array))) - {0}
    images.append({'name': path.name, 'shape': array.shape, 'dtype': str(array.dtype), 'maximum_id': int(array.max())})
assert len(images) == 2
result = {'warmup_ssl_max_abs': max(warm_diffs.values()),
          'joint_lora_changed_tensors': sum(v > 0 for v in joint_diffs.values()),
          'joint_lora_max_abs': max(joint_diffs.values()), 'native_pngs': images}
(root / 'preflight_audit.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
