"""Графики для отчёта: scatter-графики и heatmap корреляций (brainsize.txt)."""
import matplotlib.pyplot as plt
import seaborn as sns

from correlation_tasks import FEATURES, TARGET, load_brain_data


def main(csv_path: str = "data/brainsize.txt") -> None:
    df = load_brain_data(csv_path)

    # Scatter-графики: каждый признак против MRI_Count, цветом — пол
    fig, axes = plt.subplots(1, len(FEATURES), figsize=(20, 4))
    for ax, feature in zip(axes, FEATURES):
        sns.scatterplot(data=df, x=feature, y=TARGET, hue="Gender", ax=ax)
        ax.set_title(f"{feature} vs {TARGET}")
    fig.tight_layout()
    fig.savefig("scatter_mri.png", dpi=120)

    # Heatmap корреляций отдельно для мужчин и женщин
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, gender in zip(axes, ["Female", "Male"]):
        corr = df[df["Gender"] == gender][FEATURES + [TARGET]].corr(method="pearson")
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
        ax.set_title(f"Корреляции Пирсона: {gender}")
    fig.tight_layout()
    fig.savefig("heatmap_corr.png", dpi=120)
    plt.show()


if __name__ == "__main__":
    main()