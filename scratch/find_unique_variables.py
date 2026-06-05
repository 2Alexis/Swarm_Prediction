import os
import pandas as pd

raw_dir = r"c:\Users\alexi\Desktop\Swarm_Prediction\data\raw"
output_file = r"c:\Users\alexi\Desktop\Swarm_Prediction\scratch\faostat_indicators_all.txt"

faostat_files = []
for root, dirs, files in os.walk(os.path.join(raw_dir, "faostat")):
    for f in files:
        if f.endswith('.csv'):
            faostat_files.append(os.path.relpath(os.path.join(root, f), raw_dir))

with open(output_file, 'w', encoding='utf-8') as out:
    for rel_path in sorted(faostat_files):
        full_path = os.path.join(raw_dir, rel_path)
        out.write("=" * 80 + "\n")
        out.write(f"File: {rel_path}\n")
        
        # skip large files like commerce_agri (2.6 GB) or production_agricole (558 MB) or bilans_alimentaires (682 MB) unless they could be relevant
        # actually, let's look at their size.
        size_mb = os.path.getsize(full_path) / (1024 * 1024)
        out.write(f"Size: {size_mb:.2f} MB\n")
        if size_mb > 150:
            out.write("  SKIPPED scanning products/elements (File too large for fast scan, but headers printed)\n")
            # just print header and first line
            for enc in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    with open(full_path, 'r', encoding=enc) as f:
                        h = f.readline().strip()
                        l1 = f.readline().strip()
                    out.write(f"  Header ({enc}): {h[:200]}\n")
                    out.write(f"  Line 1 ({enc}): {l1[:200]}\n")
                    break
                except Exception as e:
                    continue
            continue
            
        encodings = ['utf-8', 'latin-1', 'cp1252']
        success = False
        for enc in encodings:
            try:
                # Read first chunk to detect columns
                df_first = pd.read_csv(full_path, nrows=5, encoding=enc)
                prod_col = None
                elem_col = None
                month_col = None
                for col in df_first.columns:
                    if 'produit' in col.lower() or 'item' in col.lower():
                        prod_col = col
                    if 'élément' in col.lower() or 'element' in col.lower():
                        elem_col = col
                    if 'mois' in col.lower() or 'month' in col.lower():
                        month_col = col
                
                products = set()
                elements = set()
                months = set()
                chunk_size = 50000
                cols_to_use = [c for c in [prod_col, elem_col, month_col] if c is not None]
                
                for chunk in pd.read_csv(full_path, chunksize=chunk_size, usecols=cols_to_use, encoding=enc):
                    if prod_col:
                        products.update(chunk[prod_col].dropna().unique())
                    if elem_col:
                        elements.update(chunk[elem_col].dropna().unique())
                    if month_col:
                        months.update(chunk[month_col].dropna().unique())
                
                out.write(f"  Detected columns: Product: {prod_col}, Element: {elem_col}, Month: {month_col} (encoding: {enc})\n")
                out.write(f"  Unique Products ({len(products)}):\n")
                for p in sorted(list(products)):
                    out.write(f"    - {p}\n")
                out.write(f"  Unique Elements ({len(elements)}):\n")
                for e in sorted(list(elements)):
                    out.write(f"    - {e}\n")
                if month_col:
                    out.write(f"  Unique Months/Periods ({len(months)}):\n")
                    for m in sorted(list(months)):
                        out.write(f"    - {m}\n")
                success = True
                break
            except Exception as e:
                last_err = e
                continue
        if not success:
            out.write(f"  Failed to read file: {str(last_err)}\n")

print("Finished scanning all FAOSTAT files. Results written to:", output_file)
