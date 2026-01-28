import csv
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(base_dir, "..", "data", "Indian_Real_Estate_Clean_Data.csv")
output_path = os.path.join(
    base_dir, "..", "data", "Indian_Real_Estate_Clean_Data_FIXED.csv"
)

print(f"Cleaning data from: {input_path}")

with (
    open(input_path, "r", encoding="utf-8", errors="replace") as infile,
    open(output_path, "w", newline="", encoding="utf-8") as outfile,
):
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    header = next(reader, None)
    if header:
        writer.writerow(header)
        expected_fields = len(header)
        print(f"Expected fields based on header: {expected_fields}")
    else:
        # Fallback if no header
        expected_fields = 13

    good_rows = 0
    bad_rows_fixed = 0
    skipped_rows = 0

    for i, row in enumerate(reader):
        if len(row) == expected_fields:
            writer.writerow(row)
            good_rows += 1
        elif len(row) > expected_fields:
            # STRATEGY: Merge the middle columns.
            # Usually strict columns are at the start (IDs) and end (Price, BHK).
            # The mess is usually in the middle (Description/Location).

            # Keep first 3 columns intact
            first_part = row[:3]
            # Keep last 5 columns intact (Price, BHK, Type, City, etc are usually at the end)
            last_part = row[-5:]

            # Merge everything in between
            middle_stuff = ",".join(row[3:-5])

            new_row = first_part + [middle_stuff] + last_part

            # Check if we hit the target length
            if len(new_row) == expected_fields:
                writer.writerow(new_row)
                bad_rows_fixed += 1
            else:
                # If it still doesn't match, we simply skip it to be safe
                skipped_rows += 1
        else:
            skipped_rows += 1

print("-" * 30)
print("Processing Complete.")
print(f"Good rows kept: {good_rows}")
print(f"Bad rows fixed: {bad_rows_fixed}")
print(f"Rows skipped: {skipped_rows}")
print(f"Saved to: {output_path}")
