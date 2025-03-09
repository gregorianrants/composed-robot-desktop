from pathlib import Path
import numpy as np

def load_ranges():
    return np.load(Path(__file__).parent/'threshold_ranges.npy')