import os
import json
import time

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from internvl.train.test_classifier import train_classifier
import pandas as pd
import wandb
import logging
from datetime import datetime
import csv


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_modality_percentage_analysis(
        model: str,
        base_output_path: str,
        test_modality: str = "derm",
        percentages: list = [0.0, 0.25, 0.5, 0.75, 1.0],
        shots_per_class: int = 1,
        batch_size: int = 192,
        meta_valid_path: str = "../../../processing/meta_pretrain_valid_local.json"
):
    """
    Run analysis of how different combinations of in-modality and out-modality percentages
    affect model performance, reusing checkpoints from previous steps.

    Args:
        model: Model path or identifier
        base_output_path: Base directory for saving experiment outputs
        test_modality: Modality to analyze (e.g., "derm")
        percentages: List of percentage values to test (must be sorted)
        shots_per_class: Number of shots per class for few-shot learning
        batch_size: Batch size for training
        meta_valid_path: Path to validation metadata
    """
    # Create results directory
    model_dir_str = model.split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(base_output_path, f"{model_dir_str}_{test_modality}_analysis_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)

    # Initialize results matrix
    results = np.zeros((len(percentages), len(percentages)))

    # For each out-modality percentage
    for i, out_mod_pct in enumerate(percentages):
        # Dictionary to store checkpoints for this out_mod_pct row
        last_checkpoint = None

        # For each in-modality percentage
        for j, in_mod_pct in enumerate(percentages):
            logger.info(f"\nRunning experiment with in_mod_pct={in_mod_pct}, out_mod_pct={out_mod_pct}")

            # Create experiment output directory
            exp_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_name = f"in{in_mod_pct}_out{out_mod_pct}_{exp_time}"
            output_path = os.path.join(experiment_dir, exp_name)
            os.makedirs(output_path, exist_ok=True)

            # Initialize new wandb run
            wandb.init(
                project=f"{model_dir_str}_{test_modality}_analysis",
                name=exp_name,
                config={
                    "in_mod_pct": in_mod_pct,
                    "out_mod_pct": out_mod_pct,
                    "test_modality": test_modality,
                    "shots_per_class": shots_per_class,
                    "checkpoint_from": last_checkpoint.split('/')[-2] if last_checkpoint else "None"
                },
                reinit=True
            )

            # Base number of epochs
            base_epochs = 2

            # Run training with checkpoint if available
            overall_stats = train_classifier(
                model_path=model,
                output_path=output_path,
                bs=batch_size,
                epochs=base_epochs,
                few_shot=True,
                shots_per_class=shots_per_class,
                test_modality=test_modality,
                in_mod_pct=in_mod_pct,
                out_mod_pct=out_mod_pct,
                meta_valid_path=meta_valid_path,
                eval_every=100000,
                load_checkpoint=last_checkpoint
            )

            # Get the final validation AUC for the specific modality
            val_auc = overall_stats[f'val/{test_modality}_auc']

            # Store result
            results[i, j] = val_auc

            # Save current results matrix
            np.savetxt(os.path.join(experiment_dir, 'results.csv'), results, delimiter=',')

            # Update checkpoint for next iteration
            # Find the latest checkpoint in the current output directory
            checkpoint_files = [f for f in os.listdir(output_path) if f.startswith('model_') and f.endswith('.pt')]
            if checkpoint_files:
                # Get the latest checkpoint based on step number
                latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
                last_checkpoint = os.path.join(output_path, latest_checkpoint)
            else:
                # If no intermediate checkpoints found, use the final model
                last_checkpoint = os.path.join(output_path, 'model.pt')

            wandb.finish()

            # Save intermediate results visualization
            save_results(results, percentages, experiment_dir, test_modality)

    # Generate and save final visualization
    save_results(results, percentages, experiment_dir, test_modality)

    return results


def save_results(results, percentages, output_dir, modality):
    """
    Save results matrix as heatmap and raw data.
    """
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        results,
        xticklabels=[f"{p:.2f}" for p in percentages],
        yticklabels=[f"{p:.2f}" for p in percentages],
        annot=True,
        fmt='.3f',
        cmap='viridis'
    )
    plt.xlabel('In-modality Percentage')
    plt.ylabel('Out-modality Percentage')
    plt.title(f'Validation AUC for {modality} Modality')

    # Save plot
    plt.savefig(os.path.join(output_dir, 'heatmap.png'), bbox_inches='tight', dpi=300)
    plt.close()

    # Save raw results
    np.save(os.path.join(output_dir, 'results.npy'), results)

    # Save as CSV with labels
    df = pd.DataFrame(
        results,
        columns=[f"in_mod_{p:.2f}" for p in percentages],
        index=[f"out_mod_{p:.2f}" for p in percentages]
    )
    df.to_csv(os.path.join(output_dir, 'results_matrix.csv'))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run modality percentage analysis')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Base output directory for experiments')
    parser.add_argument('--test_modality', type=str, default='derm',
                        help='Modality to analyze')
    parser.add_argument('--shots_per_class', type=int, default=1,
                        help='Number of shots per class')
    parser.add_argument('--batch_size', type=int, default=24,
                        help='Batch size for training')
    # parser.add_argument('--epochs', type=int, default=3,
    #                     help='Number of epochs for training')
    parser.add_argument('--meta_valid_path', type=str,
                        default="../../../processing/meta_pretrain_valid_local.json",
                        help='Path to validation metadata')
    parser.add_argument('--model_path', type=str, default="facebook/convnextv2-base-22k-224",
                        help='Model path for training')

    args = parser.parse_args()

    # Run analysis
    results = run_modality_percentage_analysis(
        model=args.model_path,
        base_output_path=args.output_path,
        test_modality=args.test_modality,
        shots_per_class=args.shots_per_class,
        batch_size=args.batch_size,
        meta_valid_path=args.meta_valid_path
    )