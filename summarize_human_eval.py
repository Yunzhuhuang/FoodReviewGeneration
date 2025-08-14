import argparse
import pandas as pd

METRIC_COLS = ["fluency_1to5", "coherence_1to5", "relevance_1to5", "overall_1to5"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--infile",
        type=str,
        default="human_eval_rater_scored.csv",
        help="CSV with raters' scores (same columns as the rater sheet, filled in).",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.infile)

    # Ensure required columns exist and are numeric
    for c in METRIC_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    scored = df.dropna(subset=METRIC_COLS).copy()

    # Overall summary
    overall = scored[METRIC_COLS].agg(["mean", "std", "count"]).round(3)
    overall.to_csv("human_eval_summary_overall.csv")
    print("=== Overall (mean / std / count) ===")
    print(overall)
    print()

    # By temperature
    if "temperature" in scored.columns:
        by_temp = scored.groupby("temperature")[METRIC_COLS].mean().round(3)
        by_temp.to_csv("human_eval_summary_by_temp.csv")
        print("=== Mean by temperature ===")
        print(by_temp)
        print()

    print("Wrote human_eval_summary_overall.csv and human_eval_summary_by_temp.csv")

if __name__ == "__main__":
    main()