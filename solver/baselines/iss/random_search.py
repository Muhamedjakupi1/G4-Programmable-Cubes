import numpy as np
from pathlib import Path

# __file__ is the current script's path.
# We go up 4 parent directories (....parent) to reach the project root 'G4-Programmable-Cubes'
project_root = Path(__file__).parent.parent.parent.parent

# Now, we construct the correct path down into the 'data' folder
file_path = project_root / "data" / "spoc3" / "cubes" / "ISS" / "Target_Cube_Types.npy"

# This should resolve to: C:\Users\STORM\Documents\GitHub\G4-Programmable-Cubes\data\spoc3\cubes\ISS\Initial_Config.npy
data = np.load(file_path, allow_pickle=True)

print(data)