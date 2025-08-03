
import os
import re
import argparse
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from collections import defaultdict
from pathlib import Path

def parse_logs(stat_key, target_dir, scale):
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

    log_paths = sorted(glob(f"{target_dir}/**/iter*/final_log.txt", recursive=True))

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

def plot_stats(aggregated_data_dict, stat_key, output_pdf):
    import matplotlib.pyplot as plt
    
    # Define colors for different target directories
    colors = ['mediumvioletred', 'steelblue', 'forestgreen', 'darkorange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    plt.figure(figsize=(10, 6))
    
    for idx, (target_name, aggregated) in enumerate(aggregated_data_dict.items()):
        iterations = sorted(aggregated.keys())
        values = [aggregated[i] for i in iterations]
        
        color = colors[idx % len(colors)]
        plt.plot(iterations, values, marker='o', linestyle='-', linewidth=2.5, 
                color=color, label=target_name)
    
    plt.title(f'Trend of {stat_key} score over Iterations', fontsize=16, weight='bold')
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel(f'{stat_key.capitalize()} Score', fontsize=14)
    
    # Set x-ticks to show all iterations present in any dataset
    all_iterations = set()
    for aggregated in aggregated_data_dict.values():
        all_iterations.update(aggregated.keys())
    plt.xticks(sorted(all_iterations))
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"Saved plot to {output_pdf}")

def main():
    parser = argparse.ArgumentParser(description="Plot iteration trends of audio dataset statistics.")
    parser.add_argument('--stat', type=str, default=None, choices=[None, 'mean', 'median', 'p25', 'p75', 'p95'],
                        help='Which statistic to plot (default: mean)')
    parser.add_argument('--output_dir', type=Path, default=Path("logs/plots"), help='Output directory (optional)')
    parser.add_argument('-s', '--scale', type=str, default="obj", choices=["overall", "scene", "obj"], help='Scale of the plot')
    parser.add_argument('-t', '--target_dir', type=Path, default=[Path("logs/final_logs/0118_bathroom/ceramic_mug")], help='Target directory', nargs='+')
    
    
    args = parser.parse_args()

    # Determine output directory based on scale and target directories
    if args.scale == "overall":
        output_dir = args.output_dir / "overall"
    elif args.scale == "scene":
        # If multiple target dirs from different scenes, use "multi_scene"
        scene_names = set(target_dir.parent.name for target_dir in args.target_dir)
        if len(scene_names) == 1:
            output_dir = args.output_dir / scene_names.pop()
        else:
            output_dir = args.output_dir / "multi_scene"
    else:  # obj scale
        # If multiple target dirs, use "multi_obj"
        if len(args.target_dir) == 1:
            target_dir = args.target_dir[0]
            output_dir = args.output_dir / target_dir.parent.name / target_dir.name
        else:
            output_dir = args.output_dir / "multi_obj"
    
    for stat in ['mean', 'median', 'p25', 'p75', 'p95']:
        if args.stat is None or args.stat == stat:
            # Collect aggregated data for all target directories
            aggregated_data_dict = {}
            
            for target_dir in args.target_dir:
                stats_data = parse_logs(stat, target_dir, args.scale)
                aggregated = aggregate_stats(stats_data, stat)
                # Use a meaningful name for the legend
                target_name = str(target_dir)
                
                aggregated_data_dict[target_name] = aggregated
            output_dir.mkdir(parents=True, exist_ok=True)
            plot_stats(aggregated_data_dict, stat, f'{output_dir}/stat_trend_{stat}.pdf')

if __name__ == "__main__":
    main()
