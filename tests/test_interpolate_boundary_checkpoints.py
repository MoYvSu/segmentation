import torch

from tools.interpolate_boundary_checkpoints import (
    create_interpolated_checkpoint,
    interpolate_decoder_state,
)


def test_only_boundary_tensors_are_interpolated():
    base = {
        "seg_branch.weight": torch.tensor([1.0]),
        "boundary_fpn.block.weight": torch.tensor([0.0, 2.0]),
        "boundary_branch.bias": torch.tensor([2.0]),
    }
    tuned = {
        "seg_branch.weight": torch.tensor([9.0]),
        "boundary_fpn.block.weight": torch.tensor([4.0, 6.0]),
        "boundary_branch.bias": torch.tensor([6.0]),
    }
    mixed, changed = interpolate_decoder_state(base, tuned, 0.25)
    assert torch.equal(mixed["seg_branch.weight"], torch.tensor([1.0]))
    assert torch.equal(mixed["boundary_fpn.block.weight"], torch.tensor([1.0, 3.0]))
    assert torch.equal(mixed["boundary_branch.bias"], torch.tensor([3.0]))
    assert len(changed) == 2


def test_checkpoint_uses_base_non_boundary_state():
    base = {
        "decoder_state_dict": {
            "seg_branch.weight": torch.tensor([1.0]),
            "boundary_branch.weight": torch.tensor([0.0]),
        },
        "lora_state_dict": {"lora": torch.tensor([2.0])},
        "epoch": 9,
    }
    tuned = {
        "decoder_state_dict": {
            "seg_branch.weight": torch.tensor([8.0]),
            "boundary_branch.weight": torch.tensor([4.0]),
        },
        "lora_state_dict": {"lora": torch.tensor([7.0])},
    }
    output, _ = create_interpolated_checkpoint(base, tuned, 0.5)
    assert torch.equal(output["decoder_state_dict"]["seg_branch.weight"], torch.tensor([1.0]))
    assert torch.equal(output["decoder_state_dict"]["boundary_branch.weight"], torch.tensor([2.0]))
    assert torch.equal(output["lora_state_dict"]["lora"], torch.tensor([2.0]))
    assert output["epoch"] == 9
