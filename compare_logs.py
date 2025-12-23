import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import os

def parse_log(path):
    iterations = []
    losses = []
    scores = []
    
    if not os.path.exists(path):
        print(f"[WARN] File not found: {path}")
        return iterations, losses, scores
        
    with open(path, 'r') as f:
        current_run = {'iter': [], 'loss': [], 'score': []}
        runs = [current_run]
        prev_it = -1
        
        for line in f:
            # Format: Iteration: 900, Loss: 0.0015..., Evaluation score: 33.42...
            m = re.search(r'Iteration:\s*(\d+).*Loss:\s*([0-9\.]+).*Evaluation score:\s*([0-9\.]+)', line)
            if m:
                it = int(m.group(1))
                loss = float(m.group(2))
                score = float(m.group(3))
                
                if it < prev_it:
                    current_run = {'iter': [], 'loss': [], 'score': []}
                    runs.append(current_run)
                
                current_run['iter'].append(it)
                current_run['loss'].append(loss)
                current_run['score'].append(score)
                prev_it = it
                
    return runs

def main():
    path_clean = "D:/nosaka/checkpoint/clean/logs/training_log.txt"
    path_final = "D:/nosaka/checkpoint/final/logs/training_log.txt"
    
    runs_c = parse_log(path_clean)
    runs_f = parse_log(path_final)
    
    # User Request: Plot only the first run for Clean
    if runs_c:
        runs_c = [runs_c[0]]
        
    # User Request: Limit x-axis to max iteration of Final
    max_iter_f = 0
    for run in runs_f:
        if run['iter']:
            max_iter_f = max(max_iter_f, max(run['iter']))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Plot Clean Runs
    for i, run in enumerate(runs_c):
        lbl = 'Clean (Run 1)'
        ax1.plot(run['iter'], run['loss'], label=lbl, alpha=0.7)
        ax2.plot(run['iter'], run['score'], label=lbl, alpha=0.7)

    # Plot Final Runs
    for i, run in enumerate(runs_f):
        lbl = f'Final (Run {i+1})' if len(runs_f) > 1 else 'Final'
        ax1.plot(run['iter'], run['loss'], label=lbl, alpha=0.7, linestyle='--')
        ax2.plot(run['iter'], run['score'], label=lbl, alpha=0.7, linestyle='--')

    if max_iter_f > 0:
        ax1.set_xlim(left=0, right=max_iter_f)

    ax1.set_ylabel('Loss (SmoothL1)')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_ylabel('Evaluation Score (PSNR)')
    ax2.set_xlabel('Iteration')
    ax2.set_title('Evaluation Score Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    output_path = "D:/nosaka/checkpoint/log_comparison_refined.png"
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Comparison plot saved to: {output_path}")

if __name__ == "__main__":
    main()
