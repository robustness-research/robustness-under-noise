import os
import csv
import argparse
from pathlib import Path

def analyze_colinearity_outputs(output_dir='output/rankings', base_dir: Path | None = None):
    """
    Analyze colinearity output files and check if the result is 'Colinear' or 'Not colinear'.
    Colinearity is reported per-dataset (not per dataset-model).

    Args:
        output_dir: Directory containing the output log files (relative paths are resolved from this script's folder)
        base_dir: Optional base directory to resolve relative paths from. Defaults to this script's parent folder.

    Returns:
        List of dictionaries containing dataset and colinearity status
    """
    # Map dataset -> colinear status (True/False/'Error'/None)
    dataset_status = {}

    # Resolve output directory relative to script location unless absolute
    base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parent
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = (base_dir / output_path).resolve()

    if not output_path.exists():
        print(f"Warning: Directory {output_path} does not exist")
        return []

    # Process colinearity log files: colinearity_output_{dataset}.log
    for log_file in output_path.glob('colinearity_output_*.log'):
        try:
            # Extract dataset from filename
            # Format: colinearity_output_{dataset}.log
            filename = log_file.stem  # removes .log extension
            dataset = filename.replace('colinearity_output_', '')

            # Read the file and find colinearity result
            with open(log_file, 'r') as f:
                lines = f.readlines()

            # Find the colinearity check section
            is_colinear = None

            for i, line in enumerate(lines):
                if '=== Colinearity Check for Top 3 FEAT_IMP Attributes ===' in line:
                    # Look at the next few lines after the header
                    for j in range(i + 1, min(i + 10, len(lines))):
                        next_line = lines[j].strip()
                        if '[1] "Colinear"' in next_line or next_line == 'Colinear':
                            is_colinear = True
                            break
                        elif '[1] "Not colinear"' in next_line or next_line == 'Not colinear':
                            is_colinear = False
                            break
                    break

            # Check if execution was halted (error case)
            if is_colinear is None and lines:
                last_line = lines[-1].strip()
                if 'Execution halted' in last_line:
                    is_colinear = 'Error'

            dataset_status[dataset] = is_colinear

        except Exception as e:
            print(f"Error processing {log_file.name}: {str(e)}")
            continue

    # Convert to list of dicts
    results = [{'dataset': ds, 'colinear': status} for ds, status in dataset_status.items()]
    return results


def extract_agreement_from_log(log_content):
    """
    Extract agreement statistics from ranking log content.
    
    Args:
        log_content: List of lines from the log file
        
    Returns:
        Tuple: (List of agreement rows (dicts), bool indicating if all values are NA)
    """
    agreement_data = []
    in_agreement_section = False
    
    for i, line in enumerate(log_content):
        # Look for agreement section header
        if '=== Agreement (pairwise) ===' in line:
            in_agreement_section = True
            continue
        
        # If we're in the agreement section, parse lines until we hit another section
        if in_agreement_section:
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                continue
            
            # Skip separator lines (made of dashes and spaces)
            if all(c in '-' for c in stripped.replace(' ', '')):
                continue
            
            # Stop if we hit another section
            if '===' in line:
                break
            
            # Skip header row
            if stripped.startswith('Pair') or 'Kendall' in stripped or 'Spearman' in stripped:
                continue
            
            # Parse data rows - these have fixed-width format
            # Format: "Pair                 Kendall_Tau    Spearman   IOU_TopK"
            # Data:   "FEAT_IMP vs LOCAL             NA          NA         NA"
            if ' vs ' in stripped:
                # Split on whitespace but preserve the pair information
                parts = stripped.split()
                # First 3 parts make up the pair (e.g., ["FEAT_IMP", "vs", "LOCAL"])
                if len(parts) >= 5 and parts[1] == 'vs':
                    pair = ' '.join(parts[:3])
                    kendall_tau = parts[3]
                    spearman = parts[4]
                    iou_topk = parts[5] if len(parts) > 5 else 'NA'
                    
                    agreement_data.append({
                        'pair': pair,
                        'kendall_tau': kendall_tau,
                        'spearman': spearman,
                        'iou_topk': iou_topk
                    })
    
    # Check if all agreement values are NA
    all_na = False
    if agreement_data:
        all_na = all(
            row['kendall_tau'].strip() == 'NA' and 
            row['spearman'].strip() == 'NA' and 
            row['iou_topk'].strip() == 'NA'
            for row in agreement_data
        )
    
    return agreement_data, all_na


def analyze_ranking_outputs(output_dir='output/rankings', base_dir: Path | None = None):
    """
    Analyze ranking output files and extract ranking information per dataset-method pair.

    Args:
        output_dir: Directory containing the output log files (relative paths are resolved from this script's folder)
        base_dir: Optional base directory to resolve relative paths from. Defaults to this script's parent folder.

    Returns:
        List of dictionaries containing dataset, method, ranking status, and agreement data
    """
    ranking_results = []

    # Resolve output directory relative to script location unless absolute
    base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parent
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = (base_dir / output_path).resolve()

    if not output_path.exists():
        print(f"Warning: Directory {output_path} does not exist")
        return []

    # Define known datasets and methods for proper parsing
    known_datasets = {
        "analcatdata_authorship", "badges2", "banknote", "blood-transfusion-service-center", "breast-w",
        "cardiotocography", "climate-model-simulation-crashes", "cmc", "credit-g", "diabetes",
        "eucalyptus", "iris", "kc1", "liver-disorders", "mfeat-factors",
        "mfeat-karhunen", "mfeat-zernike", "ozone-level-8hr", "pc4", "phoneme",
        "qsar-biodeg", "tic-tac-toe", "vowel", "waveform-5000", "wdbc", "wilt"
    }
    known_methods = {
        "C5.0", "ctree", "fda", "gbm", "gcvEarth", "JRip", "lvq", "mlpML", "multinom", "naive_bayes",
        "PART", "rbfDDA", "rda", "rf", "rpart", "simpls", "svmLinear", "svmRadial", "rfRules", "knn", "bayesglm"
    }

    # Process ranking log files: ranking_output_{dataset}_{method}.log
    for log_file in output_path.glob('ranking_output_*.log'):
        try:
            # Extract dataset and method from filename
            # Format: ranking_output_{dataset}_{method}.log
            filename = log_file.stem  # removes .log extension
            name = filename.replace('ranking_output_', '')

            # Try to match against known datasets to find where dataset ends and method begins
            dataset = None
            method = None
            
            # Split by hyphens and underscores to create candidate components
            # Check if any known dataset matches a prefix of the remaining name
            for known_ds in sorted(known_datasets, key=len, reverse=True):
                if name.startswith(known_ds):
                    remainder = name[len(known_ds):]
                    if remainder.startswith('_'):
                        potential_method = remainder[1:]  # Remove leading underscore
                        if potential_method in known_methods:
                            dataset = known_ds
                            method = potential_method
                            break

            if dataset is None or method is None:
                print(f"Warning: Skipping file with unexpected format: {log_file.name}")
                continue

            # Read the file and check if execution completed
            with open(log_file, 'r') as f:
                lines = f.readlines()

            # Check if execution was halted (error case)
            status = 'completed'
            if lines:
                last_line = lines[-1].strip()
                if 'Execution halted' in last_line:
                    status = 'error'

            # Extract agreement data
            agreement_data = []
            all_na = False
            if status == 'completed':
                agreement_data, all_na = extract_agreement_from_log(lines)
                # If all agreement values are NA, mark as error
                if all_na and agreement_data:
                    status = 'error'

            ranking_results.append({
                'dataset': dataset,
                'method': method,
                'status': status,
                'agreement': agreement_data
            })

        except Exception as e:
            print(f"Error processing {log_file.name}: {str(e)}")
            continue

    return ranking_results

def export_to_csv(results, output_file='ranking_colinearity_results.csv', base_dir: Path | None = None):
    """
    Export dataset-level results to CSV file. If file exists, merge with existing by dataset.

    Args:
        results: List of dictionaries with dataset and colinear status
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

    fieldnames = ['dataset', 'colinear']
    file_exists = os.path.isfile(output_path)

    # Load existing records if file exists
    existing_records = {}
    if file_exists:
        with open(output_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                key = row['dataset']
                # Keep original string values; we'll write True/False/Error/None as-is
                existing_records[key] = {'dataset': row['dataset'], 'colinear': row.get('colinear')}
        print(f"Found existing file with {len(existing_records)} records")

    # Update existing records with new results (dataset-level)
    for result in results:
        key = result['dataset']
        existing_records[key] = {'dataset': key, 'colinear': result['colinear']}

    # Write all records (existing + new) to file
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in existing_records.values():
            writer.writerow(record)

    print(f"Results exported to {output_path}")
    print(f"Total records: {len(existing_records)}")

def export_agreement_to_csv(ranking_results, output_file='ranking_agreement_results.csv', base_dir: Path | None = None):
    """
    Export dataset-method-level agreement data to CSV file.

    Args:
        ranking_results: List of dictionaries from analyze_ranking_outputs
        output_file: Output CSV filename (relative paths are resolved from this script's folder)
        base_dir: Optional base directory to resolve relative paths from. Defaults to this script's parent folder.
    """
    if not ranking_results:
        print("No ranking results to export")
        return

    # Resolve output file relative to script location unless absolute
    base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parent
    output_path = Path(output_file)
    if not output_path.is_absolute():
        output_path = (base_dir / output_path).resolve()

    fieldnames = ['dataset', 'method', 'pair', 'kendall_tau', 'spearman', 'iou_topk']
    file_exists = os.path.isfile(output_path)

    # Load existing records if file exists
    existing_records = {}
    if file_exists:
        with open(output_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                key = (row['dataset'], row['method'], row['pair'])
                existing_records[key] = row
        print(f"Found existing agreement file with {len(existing_records)} records")

    # Add new agreement data
    for result in ranking_results:
        if result['status'] == 'completed' and result['agreement']:
            dataset = result['dataset']
            method = result['method']
            for agreement_row in result['agreement']:
                key = (dataset, method, agreement_row['pair'])
                existing_records[key] = {
                    'dataset': dataset,
                    'method': method,
                    'pair': agreement_row['pair'],
                    'kendall_tau': agreement_row['kendall_tau'],
                    'spearman': agreement_row['spearman'],
                    'iou_topk': agreement_row['iou_topk']
                }

    # Write all records to file
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(existing_records.values(), key=lambda r: (r['dataset'], r['method'], r['pair'])):
            writer.writerow(record)

    print(f"Agreement results exported to {output_path}")
    print(f"Total agreement records: {len(existing_records)}")


def main():
    parser = argparse.ArgumentParser(description='Analyze ranking and colinearity logs from output files.')
    parser.add_argument('--output-dir', default='output/rankings', help='Directory containing the output log files')
    parser.add_argument('--colinearity-csv', default='colinearity_results.csv', help='Path to write colinearity results')
    parser.add_argument('--ranking-csv', default='ranking_results.csv', help='Path to write ranking results')
    parser.add_argument('--agreement-csv', default='ranking_agreement_results.csv', help='Path to write ranking agreement results')
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    # Analyze colinearity outputs
    print("=== Analyzing colinearity outputs ===")
    colinearity_results = analyze_colinearity_outputs(output_dir=args.output_dir, base_dir=base_dir)
    
    # Export colinearity to CSV
    export_to_csv(colinearity_results, output_file=args.colinearity_csv, base_dir=base_dir)

    # Print colinearity summary
    if colinearity_results:
        colinear_count = sum(1 for r in colinearity_results if r['colinear'] is True)
        not_colinear_count = sum(1 for r in colinearity_results if r['colinear'] is False)
        error_count = sum(1 for r in colinearity_results if r['colinear'] == 'Error')
        unfinished_count = sum(1 for r in colinearity_results if r['colinear'] is None)

        print("\n== Colinearity Summary (dataset-level) ==")
        print(f"Total datasets: {len(colinearity_results)}")
        print(f"Colinear datasets: {colinear_count}")
        print(f"Not colinear datasets: {not_colinear_count}")
        print(f"Errored datasets: {error_count}")
        print(f"Datasets with no result: {unfinished_count}")

        # List datasets by status
        colinear_ds = sorted(r['dataset'] for r in colinearity_results if r['colinear'] is True)
        not_colinear_ds = sorted(r['dataset'] for r in colinearity_results if r['colinear'] is False)
        errored_ds = sorted(r['dataset'] for r in colinearity_results if r['colinear'] == 'Error')
        unknown_ds = sorted(r['dataset'] for r in colinearity_results if r['colinear'] is None)

        print("\n== Datasets reported as Colinear ==")
        print("\n".join(colinear_ds) if colinear_ds else "None")

        print("\n== Datasets reported as Not colinear ==")
        print("\n".join(not_colinear_ds) if not_colinear_ds else "None")

        print("\n== Datasets with colinearity errors ==")
        print("\n".join(errored_ds) if errored_ds else "None")

        print("\n== Datasets with no colinearity result ==")
        print("\n".join(unknown_ds) if unknown_ds else "None")
    else:
        print("No colinearity logs found.")

    # Analyze ranking outputs
    print("\n=== Analyzing ranking outputs ===")
    ranking_results = analyze_ranking_outputs(output_dir=args.output_dir, base_dir=base_dir)
    
    # Export agreement to CSV
    export_agreement_to_csv(ranking_results, output_file=args.agreement_csv, base_dir=base_dir)
    
    if ranking_results:
        print(f"\nTotal ranking outputs processed: {len(ranking_results)}")
        error_count = sum(1 for r in ranking_results if r['status'] == 'error')
        completed_count = len(ranking_results) - error_count
        print(f"Completed: {completed_count}")
        print(f"Errored: {error_count}")
        
        # Group by dataset
        datasets_with_results = {}
        for r in ranking_results:
            ds = r['dataset']
            if ds not in datasets_with_results:
                datasets_with_results[ds] = {'completed': 0, 'errored': 0}
            if r['status'] == 'completed':
                datasets_with_results[ds]['completed'] += 1
            else:
                datasets_with_results[ds]['errored'] += 1
        
        print("\n== Rankings by dataset ==")
        for ds in sorted(datasets_with_results.keys()):
            stats = datasets_with_results[ds]
            print(f"{ds}: {stats['completed']} completed, {stats['errored']} errored")
        
        # Print processed vs error section with dataset-method pairs
        processed_results = [r for r in ranking_results if r['status'] == 'completed']
        errored_results = [r for r in ranking_results if r['status'] == 'error']
        
        if processed_results:
            print("\n== Successfully Processed (dataset-method) ==")
            for result in sorted(processed_results, key=lambda r: (r['dataset'], r['method'])):
                print(f"Processed: {result['dataset']}, {result['method']}")
        
        if errored_results:
            print("\n== Ranking Errors (dataset-method) ==")
            for result in sorted(errored_results, key=lambda r: (r['dataset'], r['method'])):
                print(f"Error: {result['dataset']}, {result['method']}")
    else:
        print("No ranking logs found.")


if __name__ == '__main__':
    main()