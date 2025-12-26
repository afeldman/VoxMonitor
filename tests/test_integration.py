#!/usr/bin/env python
"""Integration test for VoxMonitor × DeepSuite.

Validates:
- BaseModule inheritance
- Registry integration
- Config loading
- Trainer instantiation
"""

import sys
from pathlib import Path

def test_basemodule_inheritance():
    """Test VoxMonitorLightningModule inherits from BaseModule."""
    try:
        from deepsuite.lightning_base import BaseModule
        from voxmonitor.lightning import VoxMonitorLightningModule
        
        assert issubclass(VoxMonitorLightningModule, BaseModule)
        print("✅ BaseModule inheritance: OK")
        return True
    except Exception as e:
        print(f"❌ BaseModule inheritance: FAILED - {e}")
        return False


def test_registry_integration():
    """Test VoxMonitor registers in DeepSuite HeadRegistry."""
    try:
        from deepsuite.registry import HeadRegistry
        from voxmonitor.registry import register_voxmonitor_head
        
        success = register_voxmonitor_head()
        if success:
            key = "voxmonitor.audio.multitask"
            assert key in HeadRegistry._registry
            print("✅ Registry integration: OK")
            return True
        else:
            print("❌ Registry integration: Registration returned False")
            return False
    except Exception as e:
        print(f"❌ Registry integration: FAILED - {e}")
        return False


def test_config_loading():
    """Test config loading with new DeepSuite fields."""
    try:
        import yaml
        
        config_path = Path(__file__).parent / "config" / "config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Check new fields exist
        assert "mlflow_experiment" in config["train"]
        assert "export_formats" in config["train"]
        
        print("✅ Config loading: OK")
        return True
    except Exception as e:
        print(f"❌ Config loading: FAILED - {e}")
        return False


def test_trainer_instantiation():
    """Test BaseTrainer instantiation."""
    try:
        from deepsuite.lightning_base import BaseTrainer
        
        trainer = BaseTrainer(
            log_dir="/tmp/test_logs",
            model_output_dir="/tmp/test_models",
            export_formats=[],  # No exports for test
        )
        
        assert trainer is not None
        print("✅ Trainer instantiation: OK")
        return True
    except Exception as e:
        print(f"❌ Trainer instantiation: FAILED - {e}")
        return False


def test_module_instantiation():
    """Test VoxMonitorLightningModule instantiation."""
    try:
        from voxmonitor.lightning import VoxMonitorLightningModule
        
        module = VoxMonitorLightningModule(
            num_classes={"age": 4, "sex": 2, "valence": 3, "context": 5},
            lr=1e-3,
        )
        
        assert module is not None
        assert hasattr(module, "model")
        assert hasattr(module, "num_classes")
        
        print("✅ Module instantiation: OK")
        return True
    except Exception as e:
        print(f"❌ Module instantiation: FAILED - {e}")
        return False


def main():
    """Run all integration tests."""
    print("=" * 60)
    print("VoxMonitor × DeepSuite Integration Tests")
    print("=" * 60)
    
    tests = [
        test_basemodule_inheritance,
        test_registry_integration,
        test_config_loading,
        test_trainer_instantiation,
        test_module_instantiation,
    ]
    
    results = [test() for test in tests]
    
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed!")
        return 0
    else:
        print(f"❌ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
