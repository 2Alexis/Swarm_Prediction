from PIL import Image
import numpy as np

tif_path = r"c:\Users\alexi\Desktop\Swarm_Prediction\data\raw\WorldClim\biovar\wc2.1_10m_bio_1.tif"

try:
    img = Image.open(tif_path)
    print("Format:", img.format)
    print("Size:", img.size)
    print("Mode:", img.mode)
    
    # Load as numpy array
    arr = np.array(img)
    print("Array shape:", arr.shape)
    print("Array min/max:", arr.min(), arr.max())
    print("Array dtype:", arr.dtype)
except Exception as e:
    print("Error reading TIF:", str(e))
