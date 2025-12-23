import re

path = "D:/nosaka/checkpoint/clean/logs/training_log.txt"
prev_iter = -1
resets = []

with open(path, 'r') as f:
    for line in f:
        m = re.search(r'Iteration:\s*(\d+)', line)
        if m:
            curr_iter = int(m.group(1))
            if curr_iter < prev_iter:
                print(f"Reset detected: {prev_iter} -> {curr_iter}")
                resets.append(curr_iter)
            prev_iter = curr_iter

if not resets:
    print("No iteration resets found.")
else:
    print(f"Total resets: {len(resets)}")
