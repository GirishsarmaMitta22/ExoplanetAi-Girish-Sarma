import subprocess
import sys

with open('requirements.txt', 'w', encoding='utf-8') as f:
    subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=f, text=True)
print("Generated requirements.txt")
