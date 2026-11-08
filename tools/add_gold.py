"""Add or replace one row in gold/gold_promos.csv (used while he dictates his picks).

Usage:  .venv\\Scripts\\python tools/add_gold.py <id> "<segments>" "<hook>" "<cliffhanger>" "<rating or empty>" "<notes>"
"""
import csv
import os
import sys

PATH = os.path.join(os.path.dirname(__file__), "..", "gold", "gold_promos.csv")
COLS = ["promo_id", "segments", "hook", "cliffhanger", "rating_1_to_5", "notes"]

row = dict(zip(COLS, sys.argv[1:7]))
rows = list(csv.DictReader(open(PATH, encoding="utf-8", newline=""))) if os.path.exists(PATH) else []
rows = [r for r in rows if r["promo_id"] != row["promo_id"]] + [row]
rows.sort(key=lambda r: r["promo_id"])
with open(PATH, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS)
    w.writeheader()
    w.writerows(rows)
print(f"{row['promo_id']} saved; {len(rows)} rows in the file")
