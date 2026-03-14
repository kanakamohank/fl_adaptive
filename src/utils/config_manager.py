import yaml
import torch
import numpy as np
import random
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json


class ConfigManager:
    """
    Configuration manager for reproducible experiments.

    Handles loading configurations, setting random seeds,
    and ensuring reproducible experiment setup.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = {}

        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        logging.info(f"Configuration loaded from {config_path}")
        return self.config

    def save_config(self, save_path: str):
        """Save current configuration to file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as YAML
        with open(save_path.with_suffix('.yaml'), 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)

        # Also save as JSON for easier parsing
        with open(save_path.with_suffix('.json'), 'w') as f:
            json.dump(self.config, f, indent=2)

        logging.info(f"Configuration saved to {save_path}")

    def setup_reproducibility(self, seed: Optional[int] = None) -> int:
        """
        Set up reproducible environment with fixed random seeds.

        Args:
            seed: Random seed to use. If None, uses config value or default.

        Returns:
            The seed that was set
        """
        if seed is None:
            seed = self.get('global.seed', 42)

        # Set Python random seed
        random.seed(seed)

        # Set NumPy random seed
        np.random.seed(seed)

        # Set PyTorch random seeds
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Make PyTorch deterministic (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Set environment variable for Python hash randomization
        os.environ['PYTHONHASHSEED'] = str(seed)

        logging.info(f"Reproducibility setup complete with seed {seed}")
        return seed

    def setup_logging(self):
        """Set up logging based on configuration."""
        log_config = self.get('logging', {})

        level = getattr(logging, log_config.get('level', 'INFO').upper())
        format_str = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Configure basic logging
        logging.basicConfig(
            level=level,
            format=format_str,
            handlers=[logging.StreamHandler()]
        )

        # Add file handler if specified
        if 'file' in log_config:
            log_file = Path(log_config['file'])
            log_file.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(logging.Formatter(format_str))

            # Add to root logger
            logging.getLogger().addHandler(file_handler)

        logging.info("Logging configuration complete")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key: Configuration key in dot notation (e.g., 'global.seed')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation.

        Args:
            key: Configuration key in dot notation
            value: Value to set
        """
        keys = key.split('.')
        config_dict = self.config

        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in config_dict:
                config_dict[k] = {}
            config_dict = config_dict[k]

        # Set the value
        config_dict[keys[-1]] = value

    def get_experiment_config(self, experiment_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific experiment.

        Args:
            experiment_name: Name of the experiment (e.g., 'E1', 'E3')

        Returns:
            Experiment configuration with global values merged in
        """
        # Get base configuration
        base_config = {
            'seed': self.get('global.seed'),
            'device': self.get('global.device'),
            'results_dir': self.get('global.results_dir'),
            'data_dir': self.get('global.data_dir'),
            'model_type': self.get('model.type'),
            'model_kwargs': self.get('model.kwargs', {}),
        }

        # Get experiment-specific configuration
        exp_config = self.get(f'experiments.{experiment_name}', {})

        # Merge configurations (experiment-specific overrides base)
        merged_config = {**base_config, **exp_config}

        return merged_config

    def create_experiment_variants(self, base_experiment: str,
                                 variants: Dict[str, Dict[str, Any]]) -> Dict[str, Dict]:
        """
        Create multiple experiment configurations with parameter variations.

        Args:
            base_experiment: Name of base experiment
            variants: Dictionary of variant name -> parameter overrides

        Returns:
            Dictionary of variant configurations
        """
        base_config = self.get_experiment_config(base_experiment)

        variant_configs = {}
        for variant_name, overrides in variants.items():
            variant_config = base_config.copy()
            variant_config.update(overrides)
            variant_config['variant_name'] = variant_name
            variant_configs[variant_name] = variant_config

        return variant_configs

    def validate_config(self) -> bool:
        """
        Validate configuration completeness and correctness.

        Returns:
            True if configuration is valid
        """
        required_keys = [
            'global.seed',
            'global.device',
            'model.type',
            'data.dataset'
        ]

        missing_keys = []
        for key in required_keys:
            if self.get(key) is None:
                missing_keys.append(key)

        if missing_keys:
            logging.error(f"Missing required configuration keys: {missing_keys}")
            return False

        # Validate device
        device = self.get('global.device')
        if device == 'cuda' and not torch.cuda.is_available():
            logging.warning("CUDA requested but not available, falling back to CPU")
            self.set('global.device', 'cpu')
        elif device == 'mps' and not torch.backends.mps.is_available():
            logging.warning("MPS requested but not available, falling back to CPU")
            self.set('global.device', 'cpu')
        elif device == 'auto':
            # Auto-select best available device
            if torch.backends.mps.is_available():
                self.set('global.device', 'mps')
                logging.info("Auto-selected MPS device (Apple Silicon)")
            elif torch.cuda.is_available():
                self.set('global.device', 'cuda')
                logging.info("Auto-selected CUDA device")
            else:
                self.set('global.device', 'cpu')
                logging.info("Auto-selected CPU device")

        # Validate paths
        results_dir = Path(self.get('global.results_dir'))
        results_dir.mkdir(parents=True, exist_ok=True)

        data_dir = Path(self.get('global.data_dir'))
        data_dir.mkdir(parents=True, exist_ok=True)

        logging.info("Configuration validation complete")
        return True

    def get_full_config(self) -> Dict[str, Any]:
        """Get the complete configuration dictionary."""
        return self.config.copy()

    def print_config_summary(self):
        """Print a summary of key configuration parameters."""
        print("=" * 60)
        print("ZERO-TRUST PROJECTION FL CONFIGURATION SUMMARY")
        print("=" * 60)

        print(f"Global Settings:")
        print(f"  Seed: {self.get('global.seed')}")
        print(f"  Device: {self.get('global.device')}")
        print(f"  Results Directory: {self.get('global.results_dir')}")

        print(f"\nModel Configuration:")
        print(f"  Type: {self.get('model.type')}")
        print(f"  Parameters: {self.get('model.kwargs')}")

        print(f"\nData Configuration:")
        print(f"  Dataset: {self.get('data.dataset')}")
        print(f"  Clients: {self.get('data.num_clients')}")
        print(f"  Distribution: {self.get('data.data_distribution')} (α={self.get('data.alpha')})")

        print(f"\nProjection Settings:")
        print(f"  k_ratio: {self.get('projection.k_ratio')}")
        print(f"  Structured: {self.get('projection.structured')}")
        print(f"  Ephemeral: {self.get('projection.ephemeral')}")

        print(f"\nDetection Settings:")
        print(f"  Method: {self.get('detection.method')}")
        print(f"  Threshold: {self.get('detection.threshold')}")

        print("=" * 60)


class ExperimentTracker:
    """Track and log experiment execution for reproducibility."""

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.execution_log = []
        self.current_experiment = None

    def start_experiment(self, experiment_name: str, config: Dict[str, Any]):
        """Start tracking a new experiment."""
        import time

        self.current_experiment = {
            'name': experiment_name,
            'config': config,
            'start_time': time.time(),
            'events': []
        }

        logging.info(f"Started experiment: {experiment_name}")

    def log_event(self, event_type: str, message: str, data: Optional[Dict] = None):
        """Log an event during experiment execution."""
        import time

        if self.current_experiment is None:
            logging.warning("No active experiment to log event to")
            return

        event = {
            'timestamp': time.time(),
            'type': event_type,
            'message': message,
            'data': data
        }

        self.current_experiment['events'].append(event)
        logging.info(f"[{event_type}] {message}")

    def finish_experiment(self, results: Dict[str, Any]):
        """Finish tracking current experiment."""
        import time

        if self.current_experiment is None:
            logging.warning("No active experiment to finish")
            return

        self.current_experiment['end_time'] = time.time()
        self.current_experiment['duration'] = (
            self.current_experiment['end_time'] - self.current_experiment['start_time']
        )
        self.current_experiment['results_summary'] = results

        # Save experiment log
        log_file = self.results_dir / f"{self.current_experiment['name']}_execution_log.json"
        with open(log_file, 'w') as f:
            json.dump(self.current_experiment, f, indent=2, default=str)

        # Add to overall execution log
        self.execution_log.append(self.current_experiment.copy())

        logging.info(f"Finished experiment: {self.current_experiment['name']} "
                    f"(Duration: {self.current_experiment['duration']:.2f}s)")

        self.current_experiment = None

    def save_execution_summary(self):
        """Save summary of all executed experiments."""
        summary_file = self.results_dir / "execution_summary.json"

        summary = {
            'total_experiments': len(self.execution_log),
            'total_duration': sum(exp['duration'] for exp in self.execution_log),
            'experiments': self.execution_log
        }

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logging.info(f"Execution summary saved to {summary_file}")


def create_default_config() -> Dict[str, Any]:
    """Create a default configuration for testing."""
    return {
        'global': {
            'seed': 42,
            'device': 'cpu',
            'results_dir': './results',
            'data_dir': './data'
        },
        'model': {
            'type': 'cifar_cnn',
            'kwargs': {'num_classes': 10}
        },
        'data': {
            'dataset': 'cifar10',
            'num_clients': 20,
            'data_distribution': 'dirichlet',
            'alpha': 0.5
        },
        'experiments': {
            'E4': {
                'num_clients': 20,
                'k_values': [10, 20, 50],
                'attack_ratios': [0.0, 0.1, 0.2],
                'num_trials': 2
            }
        }
    }