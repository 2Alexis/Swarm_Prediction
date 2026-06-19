import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('clean_data.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Total lines in clean_data.py: {len(lines)}")

for idx, line in enumerate(lines):
    if any(q in line.lower() for q in ['centroid', 'physical_features', 'pays', 'coord']):
        print(f"Line {idx+1}: {line.strip()}")
