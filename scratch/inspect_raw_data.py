import os

raw_dir = r"c:\Users\alexi\Desktop\Swarm_Prediction\data\raw"
output_file = r"c:\Users\alexi\Desktop\Swarm_Prediction\scratch\inspection_results.txt"

csv_files = []
for root, dirs, files in os.walk(raw_dir):
    for f in files:
        if f.endswith('.csv'):
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, raw_dir)
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            csv_files.append((rel_path, full_path, size_mb))

with open(output_file, 'w', encoding='utf-8') as out:
    out.write(f"Found {len(csv_files)} CSV files.\n")
    for rel_path, full_path, size in sorted(csv_files):
        out.write("-" * 80 + "\n")
        out.write(f"File: {rel_path} ({size:.2f} MB)\n")
        try:
            # Read the first 3 lines as raw text
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [f.readline().strip() for _ in range(3)]
            for i, line in enumerate(lines):
                out.write(f"  Line {i+1}: {line[:300]}\n")
        except Exception as e:
            out.write(f"  Error reading file: {str(e)}\n")

print(f"Results written to {output_file}")
