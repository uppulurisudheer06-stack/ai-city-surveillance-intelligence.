from pathlib import Path

folders = [
    Path("fire_smoke_clean/train/labels"),
    Path("fire_smoke_clean/valid/labels"),
    Path("fire_smoke_clean/test/labels")
]

processed = 0

for folder in folders:
    print(f"\nChecking: {folder}")

    if not folder.exists():
        print("Folder not found")
        continue

    for file in folder.glob("*.txt"):
        lines = file.read_text().splitlines()
        new_lines = []

        for line in lines:
            parts = line.split()

            if len(parts) < 5:
                continue

            class_id = parts[0]

            try:
                values = [float(x) for x in parts[1:]]
            except ValueError:
                continue

            # Already bounding-box format
            if len(values) == 4:
                new_lines.append(line)

            # Segmentation format
            elif len(values) >= 6 and len(values) % 2 == 0:
                xs = values[0::2]
                ys = values[1::2]

                xmin = min(xs)
                xmax = max(xs)
                ymin = min(ys)
                ymax = max(ys)

                x_center = (xmin + xmax) / 2
                y_center = (ymin + ymax) / 2
                width = xmax - xmin
                height = ymax - ymin

                new_lines.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} "
                    f"{width:.6f} {height:.6f}"
                )

        file.write_text("\n".join(new_lines) + "\n")
        processed += 1

print("\n==============================")
print("LABEL CONVERSION COMPLETE")
print("==============================")
print(f"Files processed: {processed}")