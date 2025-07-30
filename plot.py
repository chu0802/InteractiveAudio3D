
import os
import re
import argparse
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from collections import defaultdict
from pathlib import Path

def parse_logs(stat_key, scene_name, target_obj):
    action_block_re = re.compile(
        r"action: (?P<action>.+?), number of samples: (?P<num>\d+)\n"
        r"max score: (?P<max>[\d\.]+)\n"
        r"min score: (?P<min>[\d\.]+)\n"
        r"mean score: (?P<mean>[\d\.]+)\n"
        r"25th percentile: (?P<p25>[\d\.]+)\n"
        r"medium score: (?P<median>[\d\.]+)\n"
        r"75th percentile: (?P<p75>[\d\.]+)\n"
        r"95th percentile score: (?P<p95>[\d\.]+)"
    )

    if scene_name is None:
        log_paths = sorted(glob("logs/final_logs/**/iter*/final_log.txt", recursive=True))
    elif target_obj is None:
        log_paths = sorted(glob(f"logs/final_logs/{scene_name}/*/iter*/final_log.txt", recursive=True))
    else:
        log_paths = sorted(glob(f"logs/final_logs/{scene_name}/{target_obj}/iter*/final_log.txt", recursive=True))

    stats_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for path in log_paths:
        iter_match = re.search(r"iter(\d+)", path)
        if not iter_match:
            continue
        iter_num = int(iter_match.group(1))

        with open(path, 'r') as f:
            content = f.read()
            for match in action_block_re.finditer(content):
                action = match.group('action')
                if stat_key not in match.groupdict():
                    continue
                stats_data[iter_num][action][stat_key].append(float(match.group(stat_key)))

    return stats_data

def aggregate_stats(stats_data, stat_key):
    aggregated = defaultdict(dict)
    for iter_num in sorted(stats_data.keys()):
        values = [
            stats_data[iter_num][action][stat_key]
            for action in stats_data[iter_num]
        ]
        aggregated[iter_num] = np.mean(values)
    return aggregated

def plot_stats(aggregated, stat_key, output_pdf):
    import matplotlib.pyplot as plt

    iterations = sorted(aggregated.keys())
    values = [aggregated[i] for i in iterations]

    plt.figure(figsize=(8, 5))
    plt.plot(iterations, values, marker='o', linestyle='-', linewidth=2.5, color='mediumvioletred')
    plt.title(f'Trend of {stat_key} score over Iterations', fontsize=16, weight='bold')
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel(f'{stat_key.capitalize()} Score', fontsize=14)
    plt.xticks(iterations)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_pdf)
    print(f"Saved plot to {output_pdf}")

def main():
    parser = argparse.ArgumentParser(description="Plot iteration trends of audio dataset statistics.")
    parser.add_argument('--stat', type=str, default=None, choices=[None, 'mean', 'median', 'p25', 'p75', 'p95'],
                        help='Which statistic to plot (default: mean)')
    parser.add_argument('--output_dir', type=Path, default=Path("logs/plots"), help='Output directory (optional)')
    parser.add_argument('-s', '--scene_name', type=str, default=None, help='Scene name')
    parser.add_argument('-t', '--target_obj', type=str, default=None, help='Target object')

    args = parser.parse_args()

    output_dir = (
        args.output_dir / "overall"
        if args.scene_name is None
        else (
            args.output_dir / args.scene_name
            if args.target_obj is None
            else (
                args.output_dir / args.scene_name / args.target_obj
            )
        )
    )

    output_pdf = f'{output_dir}/stat_trend_{args.stat}.pdf'

    for stat in ['mean', 'median', 'p25', 'p75', 'p95']:
        if args.stat is None or args.stat == stat:
            stats_data = parse_logs(stat, args.scene_name, args.target_obj)
            aggregated = aggregate_stats(stats_data, stat)
            output_dir.mkdir(parents=True, exist_ok=True)
            plot_stats(aggregated, stat, f'{output_dir}/stat_trend_{stat}.pdf')

if __name__ == "__main__":
    main()
