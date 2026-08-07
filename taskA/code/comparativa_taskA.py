from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

TASKA_RESULTS = Path(__file__).resolve().parents[1] / "risultati"

RESULT_FILES = {
    "Valence": "model_valence_results.txt",
    "Arousal": "model_arousal_results.txt",
    "Dominance": "model_dominance_results.txt",
}


def parse_metrics(text: str) -> tuple[float, float] | None:
    """Extracts MAE and Pearson r from a _results.txt file.
    Returns (mae, corr), or None if the expected lines aren't found."""
    lines = text.splitlines()
    mae_lines = [line for line in lines if "MAE" in line]
    corr_lines = [line for line in lines if "Pearson" in line]

    if not mae_lines or not corr_lines:
        return None

    try:
        mae = float(mae_lines[0].split("=")[1].strip())
        corr = float(corr_lines[0].split("=")[1].strip())
    except (IndexError, ValueError):
        return None

    return mae, corr


def main() -> None:
    rows: list[dict[str, float | str]] = []

    for dim, fname in RESULT_FILES.items():
        path = TASKA_RESULTS / fname
        if not path.exists():
            print(f"[WARNING] Missing results file for {dim}: {path}")
            continue

        text = path.read_text(encoding="utf-8")
        metrics = parse_metrics(text)
        if metrics is None:
            print(f"[WARNING] Unrecognized results format in {path}")
            continue

        mae, corr = metrics
        rows.append({"Dimension": dim, "MAE": mae, "Pearson r": corr})

    if not rows:
        print("No valid results found; check the files in taskA/risultati.")
        return

    df = pd.DataFrame(rows)

    print("\nTask A comparison table:\n")
    print(df.to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    df.plot(x="Dimension", y="MAE", kind="bar", ax=ax[0], color="skyblue", legend=False)
    ax[0].set_title("MAE by dimension")
    ax[0].set_ylabel("MAE")

    df.plot(x="Dimension", y="Pearson r", kind="bar", ax=ax[1], color="lightgreen", legend=False)
    ax[1].set_title("Pearson r by dimension")
    ax[1].set_ylabel("Pearson r")

    plt.tight_layout()
    out_path = TASKA_RESULTS / "comparativa_taskA.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Comparison chart saved to {out_path}")


if __name__ == "__main__":
    main()
