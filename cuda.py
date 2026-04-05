import torch
print(torch.__version__)
print(torch.version.cuda)
import subprocess
result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
print(result.stdout)
print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))