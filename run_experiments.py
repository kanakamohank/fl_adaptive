#!/usr/bin/env python3
"""
Main script for running Zero-Trust Projection Federated Learning experiments.

Usage:
    python run_experiments.py --config configs/default_config.yaml --experiments E1,E3,E4,E5
    python run_experiments.py --quick-test  # Run minimal test
    python run_experiments.py --all          # Run all experiments
"""

import argparse
import sys
import os
from pathlib import Path
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.config_manager import ConfigManager, ExperimentTracker, create_default_config
from src.evaluation.comprehensive_evaluator import ZeroTrustProjectionEvaluator


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Zero-Trust Projection Federated Learning Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiments.py --config configs/default_config.yaml
  python run_experiments.py --quick-test
  python run_experiments.py --experiments E1,E4 --device cuda
  python run_experiments.py --all --seed 123
        """
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/default_config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--experiments', '-e',
        type=str,
        help='Comma-separated list of experiments to run (E1,E3,E4,E5,attack_resistance)'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all experiments'
    )

    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run quick test with minimal configuration'
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'mps', 'auto'],
        help='Device to use for computation (auto selects best available)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility'
    )

    parser.add_argument(
        '--results-dir',
        type=str,
        help='Directory to save results'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print configuration and exit without running experiments'
    )

    return parser.parse_args()


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[logging.StreamHandler()]
    )


def create_quick_test_config() -> dict:
    """Create minimal configuration for quick testing."""
    return {
        'global': {
            'seed': 42,
            'device': 'auto',
            'results_dir': './results_quick_test',
            'data_dir': './data'
        },
        'model': {
            'type': 'cifar_cnn',
            'kwargs': {'num_classes': 10}
        },
        'data': {
            'dataset': 'cifar10',
            'num_clients': 10,
            'data_distribution': 'dirichlet',
            'alpha': 0.5
        },
        'federated_learning': {
            'num_rounds': 5,
            'local_epochs': 2,
            'batch_size': 32,
            'learning_rate': 0.01,
            'fraction_fit': 1.0,
            'fraction_evaluate': 1.0
        },
        'projection': {
            'k_ratio': 0.1,
            'structured': True,
            'ephemeral': True
        },
        'detection': {
            'method': 'isomorphic_verification',
            'threshold': 1.5,
            'min_consensus': 0.6
        },
        'experiments': {
            'E4': {  # Only run E4 for quick test
                'num_clients': 10,  # Reduced
                'k_values': [5, 10],  # Reduced
                'attack_ratios': [0.0, 0.1],  # Reduced
                'num_trials': 1  # Reduced
            }
        }
    }


def main():
    """Main execution function."""
    args = parse_arguments()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("ZERO-TRUST PROJECTION FEDERATED LEARNING")
    logger.info("=" * 60)

    try:
        # Setup configuration
        if args.quick_test:
            logger.info("Running quick test configuration")
            config_manager = ConfigManager()
            config_manager.config = create_quick_test_config()
        else:
            logger.info(f"Loading configuration from {args.config}")
            config_manager = ConfigManager(args.config)

        # Override configuration with command line arguments
        if args.device:
            config_manager.set('global.device', args.device)

        if args.seed:
            config_manager.set('global.seed', args.seed)

        if args.results_dir:
            config_manager.set('global.results_dir', args.results_dir)

        # Validate configuration
        if not config_manager.validate_config():
            logger.error("Configuration validation failed")
            return 1

        # Setup reproducibility
        seed = config_manager.setup_reproducibility()

        # Print configuration summary
        config_manager.print_config_summary()

        if args.dry_run:
            logger.info("Dry run completed - configuration validated successfully")
            return 0

        # Initialize evaluator
        evaluator = ZeroTrustProjectionEvaluator(
            results_dir=config_manager.get('global.results_dir'),
            device=config_manager.get('global.device'),
            seed=seed
        )

        # Initialize experiment tracker
        tracker = ExperimentTracker(config_manager.get('global.results_dir'))

        # Determine which experiments to run
        if args.all:
            experiments_to_run = ['E1', 'E3', 'E4', 'E5', 'attack_resistance']
        elif args.experiments:
            experiments_to_run = [exp.strip() for exp in args.experiments.split(',')]
        elif args.quick_test:
            experiments_to_run = ['E4']
        else:
            # Run all experiments if none specified
            experiments_to_run = ['E1', 'E3', 'E4', 'E5', 'attack_resistance']

        logger.info(f"Experiments to run: {experiments_to_run}")

        # Run experiments
        if len(experiments_to_run) == 1 and not args.quick_test:
            # Run single experiment
            experiment_name = experiments_to_run[0]
            exp_config = config_manager.get_experiment_config(experiment_name)

            tracker.start_experiment(experiment_name, exp_config)

            if experiment_name == 'E1':
                results = evaluator.run_e1_clean_accuracy_preservation(exp_config)
            elif experiment_name == 'E3':
                results = evaluator.run_e3_scalability_comparison(exp_config)
            elif experiment_name == 'E4':
                results = evaluator.run_e4_optimal_k_selection(exp_config)
            elif experiment_name == 'E5':
                results = evaluator.run_e5_structured_vs_unstructured(exp_config)
            elif experiment_name == 'attack_resistance':
                results = evaluator.run_comprehensive_attack_resistance_evaluation(exp_config)
            else:
                logger.error(f"Unknown experiment: {experiment_name}")
                return 1

            tracker.finish_experiment(results)

        else:
            # Run multiple experiments or comprehensive evaluation
            all_config = {
                exp: config_manager.get_experiment_config(exp)
                for exp in experiments_to_run if exp != 'attack_resistance'
            }

            if 'attack_resistance' in experiments_to_run:
                all_config['attack_resistance'] = config_manager.get_experiment_config('attack_resistance')

            tracker.start_experiment('comprehensive_evaluation', all_config)

            # Run comprehensive evaluation
            results = evaluator.run_all_experiments(all_config)

            tracker.finish_experiment(results)

        # Save execution summary
        tracker.save_execution_summary()

        # Print final summary
        logger.info("=" * 60)
        logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

        if 'final_summary' in results:
            summary = results['final_summary']
            logger.info("Key Findings:")
            for finding, details in summary.get('key_findings', {}).items():
                logger.info(f"  {finding}: {details}")

            logger.info("\nRecommendations:")
            for rec in summary.get('recommendations', []):
                logger.info(f"  - {rec}")

        logger.info(f"\nResults saved to: {config_manager.get('global.results_dir')}")

        return 0

    except KeyboardInterrupt:
        logger.warning("Experiment interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Experiment failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())