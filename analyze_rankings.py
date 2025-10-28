import os
import csv
import argparse
from pathlib import Path

def analyze_ranking_outputs(output_dir='output/rankings', base_dir: Path | None = None):
    """
    Analyze ranking output files and check if the last result is 'Colinear' or 'Not colinear'.

    Args:
        output_dir: Directory containing the output log files (relative paths are resolved from this script's folder)
        base_dir: Optional base directory to resolve relative paths from. Defaults to this script's parent folder.

    Returns:
        List of dictionaries containing dataset, model, and colinearity status
    """
    results = []

    # Resolve output directory relative to script location unless absolute
    base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parent
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = (base_dir / output_path).resolve()

    if not output_path.exists():
        print(f"Warning: Directory {output_path} does not exist")
        return results
    
    # Process all log files in the directory
    for log_file in output_path.glob('output_*.log'):
        try:
            # Extract dataset and method from filename
            # Format: output_{dataset}_{method}.log
            filename = log_file.stem  # removes .log extension
            parts = filename.replace('output_', '').rsplit('_', 1)
            
            if len(parts) != 2:
                print(f"Warning: Skipping file with unexpected format: {log_file.name}")
                continue
                
            dataset, method = parts
            
            # Read the file and find colinearity result
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            # Find the colinearity check section
            is_colinear = None
            header_found = False
            
            for i, line in enumerate(lines):
                if '=== Colinearity Check for Top 3 FEAT_IMP Attributes ===' in line:
                    header_found = True
                    # Look at the next few lines after the header
                    for j in range(i + 1, min(i + 10, len(lines))):
                        next_line = lines[j].strip()
                        if '[1] "Colinear"' in next_line:
                            is_colinear = True
                            break
                        elif '[1] "Not colinear"' in next_line:
                            is_colinear = False
                            break
                    break
            
            # Check if execution was halted (error case)
            if is_colinear is None and lines:
                last_line = lines[-1].strip()
                if 'Execution halted' in last_line:
                    is_colinear = 'Error'
            
            results.append({
                'dataset': dataset,
                'model': method,
                'colinear': is_colinear
            })
            
        except Exception as e:
            print(f"Error processing {log_file.name}: {str(e)}")
            continue
    
    return results

def export_to_csv(results, output_file='ranking_colinearity_results.csv', base_dir: Path | None = None):
    """
    Export results to CSV file. If file exists, merge with existing by (dataset, model).

    Args:
        results: List of dictionaries with dataset, model, and colinear status
        output_file: Output CSV filename (relative paths are resolved from this script's folder)
        base_dir: Optional base directory to resolve relative paths from. Defaults to this script's parent folder.
    """
    if not results:
        print("No results to export")
        return

    # Resolve output file relative to script location unless absolute
    base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parent
    output_path = Path(output_file)
    if not output_path.is_absolute():
        output_path = (base_dir / output_path).resolve()

    fieldnames = ['dataset', 'model', 'colinear']
    file_exists = os.path.isfile(output_path)

    # Load existing records if file exists
    existing_records = {}
    if file_exists:
        with open(output_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                key = (row['dataset'], row['model'])
                existing_records[key] = row
        print(f"Found existing file with {len(existing_records)} records")

    # Update existing records with new results
    for result in results:
        key = (result['dataset'], result['model'])
        existing_records[key] = result

    # Write all records (existing + new) to file
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in existing_records.values():
            writer.writerow(record)

    print(f"Results exported to {output_path}")
    print(f"Total records: {len(existing_records)}")

def main():
    parser = argparse.ArgumentParser(description='Analyze ranking logs and export colinearity results to CSV.')
    parser.add_argument('--output-dir', default='output/rankings', help='Directory containing the output log files')
    parser.add_argument('--output-csv', default='ranking_colinearity_results.csv', help='Path to write the aggregated CSV results')
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    # Analyze the outputs
    results = analyze_ranking_outputs(output_dir=args.output_dir, base_dir=base_dir)

    # Export to CSV
    export_to_csv(results, output_file=args.output_csv, base_dir=base_dir)

    # Print enhanced summary
    if results:
        colinear_count = sum(1 for r in results if r['colinear'] is True)
        not_colinear_count = sum(1 for r in results if r['colinear'] is False)
        error_count = sum(1 for r in results if r['colinear'] == 'Error')
        unfinished_count = sum(1 for r in results if r['colinear'] is None)

        print("\n== Overall summary ==")
        print(f"Total results: {len(results)}")
        print(f"Colinear: {colinear_count}")
        print(f"Not colinear: {not_colinear_count}")
        print(f"Errored: {error_count}")
        print(f"Not finished (no result): {unfinished_count}")

        # Per-dataset aggregation
        from collections import defaultdict

        ds_counts = defaultdict(lambda: {'colinear': 0, 'not_colinear': 0})
        for r in results:
            # Normalize dataset names: if dataset ends with '_naive', strip that suffix
            raw_ds = r['dataset']
            ds = raw_ds[:-6] if raw_ds.endswith('_naive') else raw_ds
            if r['colinear'] is True:
                ds_counts[ds]['colinear'] += 1
            elif r['colinear'] is False:
                ds_counts[ds]['not_colinear'] += 1

        print("\n== Per-dataset colinearity counts =")
        if ds_counts:
            for ds in sorted(ds_counts):
                c = ds_counts[ds]['colinear']
                n = ds_counts[ds]['not_colinear']
                print(f"{ds}: Colinear={c}, Not colinear={n}")
        else:
            print("No per-dataset data available.")

        # Per-model aggregation
        model_counts = defaultdict(lambda: {'colinear': 0, 'not_colinear': 0})
        for r in results:
            m = r['model']
            if r['colinear'] is True:
                model_counts[m]['colinear'] += 1
            elif r['colinear'] is False:
                model_counts[m]['not_colinear'] += 1

        print("\n== Per-model colinearity counts ==")
        if model_counts:
            for m in sorted(model_counts):
                c = model_counts[m]['colinear']
                n = model_counts[m]['not_colinear']
                print(f"{m}: Colinear={c}, Not colinear={n}")
        else:
            print("No per-model data available.")

        # Print failed-model counts (how many errors per model) and the most-failed model(s)
        from collections import Counter

        failed_models = Counter(r['model'] for r in results if r['colinear'] == 'Error')
        print("\n== Failed models (by failure count) ==")
        if failed_models:
            for model, cnt in failed_models.most_common():
                print(f"{model}: {cnt}")
            # Find the maximum failures and list all models that share that maximum
            max_cnt = failed_models.most_common(1)[0][1]
            most_failed = [m for m, c in failed_models.items() if c == max_cnt]
            print(f"\nMost failed model(s) [{max_cnt} failures]: {', '.join(sorted(most_failed))}")
        else:
            print("None")

        # Print failed combinations (dataset-model) that ended in error
        failed = [(r['dataset'], r['model']) for r in results if r['colinear'] == 'Error']
        print("\n== Failed dataset-model combinations ==")
        if failed:
            for ds, m in failed:
                print(f"- {ds} | {m}")
        else:
            print("None")
    else:
        print("No logs processed.")


if __name__ == '__main__':
    main()