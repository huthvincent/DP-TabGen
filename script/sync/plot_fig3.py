from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def plot_lambda_curve(out_path: Path) -> None:
    # lambdas from 0.00 to 0.19 (step 0.01)
    lambdas = [round(x * 0.01, 2) for x in range(0, 20)]
    accuracies = [0.781, 0.779, 0.841, 0.837, 0.872, 0.891, 0.881, 0.883, 0.871, 0.862, 0.864, 0.869, 0.865, 0.868, 0.862, 0.845, 0.840, 0.845, 0.830, 0.835]
    df = pd.DataFrame({"lambda": lambdas, "accuracy": accuracies})

    sns.set_theme(context="talk", style="whitegrid")
    plt.figure(figsize=(8, 5))
    ax = sns.lineplot(
        data=df,
        x="lambda",
        y="accuracy",
        marker="o",
        markersize=8,
        linewidth=2.2,
        color="#1b9e77",  # consistent highlight color
    )
    ax.set_xlabel("λ")
    ax.set_ylabel("Accuracy")
    # sparser ticks for readability
    tick_positions = [round(v, 2) for v in [0.00, 0.04, 0.08, 0.12, 0.16, 0.20] if v <= max(lambdas) + 1e-9]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{v:.2f}" for v in tick_positions], rotation=0, fontsize=12, fontweight="bold")
    ax.set_ylim(0.75, 0.92)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    out_file = Path(__file__).resolve().parents[2] / "plots/fig3_gain_bar.png"
    plot_lambda_curve(out_file)
