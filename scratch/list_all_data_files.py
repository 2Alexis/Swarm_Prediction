import os

for root, dirs, files in os.walk('data'):
    for file in files:
        path = os.path.join(root, file)
        size = os.path.getsize(path)
        print(f"{path} ({size:,} bytes)")
