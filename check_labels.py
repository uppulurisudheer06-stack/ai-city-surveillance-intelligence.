from pathlib import Path

root = Path("fire_smoke_clean")

total = 0
valid = 0
invalid = 0

for label_file in root.glob("**/labels/*.txt"):
    total += 1

    try:
        with open(label_file, "r") as f:
            lines = f.readlines()

        file_valid = True

        for line in lines:
            parts = line.strip().split()

            # YOLO label must have 5 values
            if len(parts) != 5:
                file_valid = False
                break

            class_id, x, y, w, h = parts

            # Check numbers
            class_id = int(class_id)
            x = float(x)
            y = float(y)
            w = float(w)
            h = float(h)

            # YOLO coordinates must be between 0 and 1
            if not (0 <= x <= 1 and
                    0 <= y <= 1 and
                    0 <= w <= 1 and
                    0 <= h <= 1):
                file_valid = False
                break

        if file_valid:
            valid += 1
        else:
            invalid += 1
            print("INVALID:", label_file)

    except Exception:
        invalid += 1
        print("ERROR:", label_file)

print("\n========== LABEL CHECK ==========")
print("Total label files :", total)
print("Valid label files :", valid)
print("Invalid files     :", invalid)

if invalid == 0:
    print("\nALL LABELS ARE VALID!")
else:
    print("\nSOME LABELS NEED FIXING!")