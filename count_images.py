from pathlib import Path

exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
root = Path("data")

total = 0

for d in [root / "old_dataset", root / "new_dataset"]:
    if d.exists():
        count = sum(1 for p in d.rglob("*") if p.is_file() and p.suffix.lower() in exts)
        print(f"{d.name}: {count}")
        total += count
    else:
        print(f"{d.name}: MISSING")

print("TOTAL:", total)