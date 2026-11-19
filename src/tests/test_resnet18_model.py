#!/usr/bin/env python3
"""ResNet-18 must be usable by THIS system, not just constructible.

The bar is not "does it run a forward pass". Three properties of this
codebase decide whether a backbone works at all, and each gets a test:

  1. TRANSPORT. Weights move between client and server as
     `zip(model.parameters(), ndarrays)` -- in the Flower client and again
     in end_to_end_pipeline's evaluate_fn. Anything held in a BUFFER never
     leaves the client. BatchNorm keeps running_mean/running_var as buffers,
     so a BatchNorm ResNet would evaluate with freshly-initialised
     statistics for the entire run and score near chance. That failure is
     silent and looks like "the architecture does not work here", so it is
     pinned first.

  2. BLOCK STRUCTURE. BVD and ESP both index `model.structure`. A block
     whose recorded size disagrees with the real tensor corrupts the
     projection that uses it.

  3. CIFAR GEOMETRY. torchvision's 7x7/stride-2 stem plus maxpool destroys a
     32x32 image before the first residual stage. The CIFAR variant used by
     FedCorr, FedGR and FNBench is a 3x3 stride-1 stem with no maxpool.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.core.models import ResNet18, get_model  # noqa: E402


def test_no_buffers_so_nothing_is_lost_in_transport():
    """THE load-bearing property. If this fails, the model is unusable here
    however good its accuracy would be offline."""
    m = ResNet18(num_classes=10)
    buffers = [n for n, _ in m.named_buffers()]
    assert buffers == [], (
        f"ResNet18 must hold NO buffers -- transport moves only "
        f"model.parameters(), so these would never reach the server: "
        f"{buffers}"
    )
    # And prove the mechanism it is avoiding, so the test explains itself.
    import torch.nn as nn
    bn = nn.BatchNorm2d(8)
    assert [n for n, _ in bn.named_buffers()], (
        "if BatchNorm ever stops holding buffers, this guard and the "
        "ResNet18 docstring are both stale"
    )
    print("✓ no buffers: every parameter is communicated")
    return True


def test_a_full_parameter_round_trip_is_exact():
    """Simulate the real transport: flatten every parameter into ndarrays,
    rebuild a fresh model, load them back, and require identical outputs.
    This is what evaluate_fn does each round."""
    torch.manual_seed(0)
    src = ResNet18(num_classes=10)
    for p in src.parameters():          # move away from init so it is a real test
        p.data.add_(torch.randn_like(p) * 0.05)
    ndarrays = [p.detach().cpu().numpy() for p in src.parameters()]

    dst = ResNet18(num_classes=10)
    for param, new in zip(dst.parameters(), ndarrays):
        param.data = torch.tensor(new)

    src.eval()
    dst.eval()
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        assert torch.allclose(src(x), dst(x), atol=0, rtol=0), (
            "a parameters()-only round trip must be exact; if it is not, "
            "some state lives outside parameters() and is being dropped"
        )
    print("✓ a parameters()-only round trip reproduces outputs exactly")
    return True


def test_block_structure_matches_the_real_tensors():
    """Every block must name a real parameter and record its true size."""
    m = ResNet18(num_classes=10)
    real = {n.replace(".", "_"): p.numel() for n, p in m.named_parameters()}
    blocks = {b["name"]: b["num_params"] for b in m.structure.blocks}
    assert blocks == real, (
        f"block structure disagrees with the model: "
        f"missing={set(real) - set(blocks)}, extra={set(blocks) - set(real)}"
    )
    assert m.structure.total_params == sum(real.values())
    assert m.structure.total_params == sum(p.numel() for p in m.parameters())
    print(f"✓ {len(blocks)} blocks, sizes match the real tensors exactly")
    return True


def test_cifar_stem_preserves_resolution():
    """A 32x32 input must still be 32x32 after the stem. torchvision's stem
    would have reduced it to 8x8 before the first residual stage."""
    m = ResNet18(num_classes=10)
    assert m.conv1.kernel_size == (3, 3), m.conv1.kernel_size
    assert m.conv1.stride == (1, 1), m.conv1.stride
    assert not any(isinstance(mod, torch.nn.MaxPool2d) for mod in m.modules()), (
        "the CIFAR variant has no maxpool in the stem"
    )
    with torch.no_grad():
        stem = torch.nn.functional.relu(m.gn1(m.conv1(torch.randn(1, 3, 32, 32))))
    assert stem.shape[-2:] == (32, 32), stem.shape
    print("✓ CIFAR stem keeps 32x32 into the first residual stage")
    return True


def test_shapes_and_size_are_right():
    m = ResNet18(num_classes=10)
    n = sum(p.numel() for p in m.parameters())
    assert 11_000_000 < n < 11_400_000, f"expected ~11.17M params, got {n:,}"
    with torch.no_grad():
        assert m(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    m100 = ResNet18(num_classes=100)
    with torch.no_grad():
        assert m100(torch.randn(2, 3, 32, 32)).shape == (2, 100)
    print(f"✓ {n:,} params; 10- and 100-class heads both work")
    return True


def test_factory_exposes_it_and_leaves_the_default_alone():
    """get_model must serve 'resnet18' AND keep 'cifar_cnn' unchanged --
    every result on disk was produced by the latter."""
    assert isinstance(get_model("resnet18", num_classes=10), ResNet18)
    from src.core.models import CIFARCNN
    assert isinstance(get_model("cifar_cnn", num_classes=10), CIFARCNN)
    try:
        get_model("nonexistent_model")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown model_type must still raise")
    print("✓ factory serves resnet18; cifar_cnn is untouched")
    return True


def test_esp_projection_memory_is_not_worse_than_today():
    """ESP builds a k_m x d_m matrix per block. More blocks and 10x the
    parameters could have exploded it, which would be a real blocker -- so
    measure rather than assume. The CIFARCNN already allocates ~7.4 GB per
    round and runs today, so ResNet-18 only has to be comparable."""
    from src.tavs_v2.algo2_esp_projection import EphemeralStructuredProjection

    def gigabytes(blocks):
        e = EphemeralStructuredProjection(target_k=2048, model_blocks=blocks,
                                          master_key=b"k")
        return sum(e.k_m_allocations[b] * blocks[b] for b in blocks) * 4 / 1e9

    from src.core.models import CIFARCNN
    cnn = {b["name"]: b["num_params"] for b in CIFARCNN().structure.blocks}
    rn = {b["name"]: b["num_params"]
          for b in ResNet18().structure.blocks}
    cnn_gb, rn_gb = gigabytes(cnn), gigabytes(rn)
    assert rn_gb < cnn_gb * 1.5, (
        f"ResNet-18 ESP matrices {rn_gb:.2f} GB vs CIFARCNN {cnn_gb:.2f} GB "
        f"-- a blow-up here blocks the switch"
    )
    print(f"✓ ESP matrices {rn_gb:.2f} GB vs the CNN's {cnn_gb:.2f} GB today")
    return True


def main():
    ok = [
        test_no_buffers_so_nothing_is_lost_in_transport(),
        test_a_full_parameter_round_trip_is_exact(),
        test_block_structure_matches_the_real_tensors(),
        test_cifar_stem_preserves_resolution(),
        test_shapes_and_size_are_right(),
        test_factory_exposes_it_and_leaves_the_default_alone(),
        test_esp_projection_memory_is_not_worse_than_today(),
    ]
    if all(ok):
        print("\n\U0001F3AF All ResNet-18 tests PASSED!")
        return True
    return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
