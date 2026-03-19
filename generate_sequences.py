"""
Batch generate sequence CSV files for Grid Puzzle experiment.

Usage:
    python generate_sequences.py

Generated files are placed in public/sequences/ as 1.csv, 2.csv, ...
Access via URL parameter: ?id=1, ?id=2, etc.
"""

import os
import random

# --- Configuration ---
NUM_FILES = 10          # Number of CSV files to generate
ROUNDS_PER_FILE = 4     # Number of rounds per file

DIFFICULTIES = [
    "Tutorial", "Beginner", "Easy", "Medium",
    "Hard", "Expert", "Master", "Grandmaster",
]

PREFILL_OPTIONS = [0, 1, 3, 5, 10, 15, 20]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "public", "sequences")
# ---------------------


def generate_csv(file_id: int) -> str:
    lines = ["Difficulty,Prefill"]
    for _ in range(ROUNDS_PER_FILE):
        diff = random.choice(DIFFICULTIES)
        prefill = random.choice(PREFILL_OPTIONS)
        lines.append(f"{diff},{prefill}")
    return "\n".join(lines) + "\n"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i in range(1, NUM_FILES + 1):
        content = generate_csv(i)
        path = os.path.join(OUTPUT_DIR, f"{i}.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Generated {path}")

    print(f"\nDone! {NUM_FILES} files written to {OUTPUT_DIR}/")
    print(f"Access in browser: ?id=1  through  ?id={NUM_FILES}")


if __name__ == "__main__":
    main()
