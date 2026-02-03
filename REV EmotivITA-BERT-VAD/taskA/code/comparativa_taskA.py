import pandas as pd
import matplotlib.pyplot as plt  # type: ignore
from pathlib import Path

"""
Legge i file di risultati per Valence, Arousal, Dominance e produce:
- una tabella comparativa in console
- un grafico comparativo salvato in taskA/risultati/comparativa_taskA.png
"""


RESULT_PATHS = {
    "Valence": "taskA/risultati/model_valence_results.txt",
    "Arousal": "taskA/risultati/model_arousal_results.txt",
    "Dominance": "taskA/risultati/model_dominance_results.txt",
}


def parse_metrics(text: str) -> tuple[float, float] | None:
    """Estrae MAE e Pearson r dal testo di un file _results.txt.
    Restituisce (mae, corr) oppure None se non trova le righe attese.
    """
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

    for dim, path_str in RESULT_PATHS.items():
        path = Path(path_str)
        if not path.exists():
            print(f"[ATTENZIONE] File risultati mancante per {dim}: {path}")
            continue

        text = path.read_text(encoding="utf-8")
        metrics = parse_metrics(text)
        if metrics is None:
            print(f"[ATTENZIONE] Formato risultati non riconosciuto in {path}")
            continue

        mae, corr = metrics
        rows.append({"Dimensione": dim, "MAE": mae, "Pearson r": corr})

    if not rows:
        print("Nessun risultato valido trovato; controlla i file in taskA/risultati.")
        return

    df = pd.DataFrame(rows)

    # --- stampa tabella in console ---
    print("\nTabella comparativa Task A:\n")
    print(df.to_string(index=False))

    # --- grafico comparativo ---
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # grafico MAE
    df.plot(x="Dimensione", y="MAE", kind="bar", ax=ax[0], color="skyblue", legend=False)
    ax[0].set_title("MAE per dimensione")
    ax[0].set_ylabel("MAE")

    # grafico Pearson r
    df.plot(x="Dimensione", y="Pearson r", kind="bar", ax=ax[1], color="lightgreen", legend=False)
    ax[1].set_title("Pearson r per dimensione")
    ax[1].set_ylabel("Pearson r")

    plt.tight_layout()
    out_path = "taskA/risultati/comparativa_taskA.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Grafico comparativo salvato in {out_path}")


if __name__ == "__main__":
    main()
