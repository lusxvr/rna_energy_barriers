import numpy as np
import matplotlib.pyplot as plt
import RNA
import time
import pandas as pd
import seaborn as sns
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import Counter
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import pearsonr, spearmanr

from src.rna_structure import RNAStructure
from src.path_finding import find_direct_path, find_best_indirect_path
from src.evolution import best_folding
import src.energy as energy


def calculate_path_metrics(sequence: str, path: List[str]) -> Dict[str, Any]:
    """
    Calculate various metrics for a given path between RNA structures.
    
    Args:
        sequence: The RNA sequence
        path: List of structures in dot-bracket notation
        
    Returns:
        Dictionary with various metrics
    """
    if not path or len(path) <= 1:
        return {
            "path_length": 0,
            "energy_barrier": 0,
            "average_energy": 0,
            "energy_profile": [],
            "initial_energy": 0,
            "final_energy": 0,
            "max_energy": 0,
            "min_energy": 0,
            "transitions": []
        }
    
    # Create ViennaRNA fold compound
    fc = RNA.fold_compound(sequence)
    
    # Calculate energy profile
    energy_profile = [energy.turner_energy(fc, struct) for struct in path]
    
    # Calculate transitions between consecutive structures
    transitions = [(path[i], path[i+1]) for i in range(len(path)-1)]
    
    # Calculate transition types
    transition_types = [classify_transition(t[0], t[1]) for t in transitions]
    
    # Count transition types
    type_counts = Counter(transition_types)
    
    return {
        "path_length": len(path) - 1,
        "energy_barrier": max(energy_profile) - energy_profile[0],
        "average_energy": np.mean(energy_profile),
        "energy_profile": energy_profile,
        "initial_energy": energy_profile[0],
        "final_energy": energy_profile[-1],
        "max_energy": max(energy_profile),
        "min_energy": min(energy_profile),
        "transitions": transitions,
        "transition_types": transition_types,
        "add_transitions": type_counts.get("add", 0),
        "remove_transitions": type_counts.get("remove", 0),
        "both_transitions": type_counts.get("both", 0)
    }


def classify_transition(struct1: str, struct2: str) -> str:
    """
    Classify the transition between two RNA structures.
    
    Args:
        struct1: First structure in dot-bracket notation
        struct2: Second structure in dot-bracket notation
        
    Returns:
        String indicating transition type: "add", "remove", or "both"
    """
    if len(struct1) != len(struct2):
        raise ValueError("Structures must have the same length")
    
    add_count = 0
    remove_count = 0
    
    for i in range(len(struct1)):
        if struct1[i] == '.' and struct2[i] in '()':
            add_count += 1
        elif struct1[i] in '()' and struct2[i] == '.':
            remove_count += 1
    
    if add_count > 0 and remove_count == 0:
        return "add"
    elif add_count == 0 and remove_count > 0:
        return "remove"
    else:
        return "both"


def compare_algorithms(sequence: str, start_struct: str, end_struct: str, 
                     methods=['direct', 'indirect', 'evolutionary'],
                     evolutionary_params={'N': 100, 'max_steps': 100, 'alpha': 0.7, 'beta': 0},
                     indirect_attempts=5) -> Dict[str, Dict[str, Any]]:
    """
    Compare different path-finding algorithms on the same RNA folding problem.
    
    Args:
        sequence: The RNA sequence
        start_struct: Starting structure in dot-bracket notation
        end_struct: Target structure in dot-bracket notation
        methods: List of methods to compare ('direct', 'indirect', 'evolutionary')
        evolutionary_params: Parameters for the evolutionary algorithm
        indirect_attempts: Number of attempts for the indirect path algorithm
        
    Returns:
        Dictionary mapping method names to their metrics
    """
    results = {}
    
    # Create RNAStructure objects
    start = RNAStructure(sequence, structure=start_struct)
    end = RNAStructure(sequence, structure=end_struct)
    
    # Direct path
    if 'direct' in methods:
        start_time = time.time()
        direct_path = find_direct_path(start, end)
        direct_path_str = [p.to_dotbracket() for p in direct_path]
        execution_time = time.time() - start_time
        
        metrics = calculate_path_metrics(sequence, direct_path_str)
        metrics['execution_time'] = execution_time
        results['direct'] = metrics
    
    # Indirect path
    if 'indirect' in methods:
        start_time = time.time()
        indirect_path = find_best_indirect_path(start, end, num_attempts=indirect_attempts)
        indirect_path_str = [p.to_dotbracket() for p in indirect_path]
        execution_time = time.time() - start_time
        
        metrics = calculate_path_metrics(sequence, indirect_path_str)
        metrics['execution_time'] = execution_time
        results['indirect'] = metrics
    
    # Evolutionary approach
    if 'evolutionary' in methods:
        start_time = time.time()
        best, steps = best_folding(
            sequence, 
            start_struct, 
            end_struct, 
            N=evolutionary_params.get('N', 100),
            max_steps=evolutionary_params.get('max_steps', 100),
            alpha=evolutionary_params.get('alpha', 0.7),
            beta=evolutionary_params.get('beta', 0)
        )
        execution_time = time.time() - start_time
        
        evolutionary_path = best['path']
        metrics = calculate_path_metrics(sequence, evolutionary_path)
        metrics['execution_time'] = execution_time
        metrics['generation_steps'] = steps
        results['evolutionary'] = metrics
    
    return results


def plot_comparison(results: Dict[str, Dict[str, Any]], figsize=(14, 10), save_path=None):
    """
    Plot a comparison of different path-finding algorithms.
    
    Args:
        results: Results from compare_algorithms
        figsize: Figure size
        save_path: Path to save the figure (if provided)
    """
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Comparison of Path-Finding Algorithms', fontsize=16)
    
    methods = list(results.keys())
    
    # Energy profiles
    ax = axs[0, 0]
    for method in methods:
        energy_profile = results[method]['energy_profile']
        steps = list(range(len(energy_profile)))
        ax.plot(steps, energy_profile, marker='o', label=method.capitalize())
    
    ax.set_title('Energy Profiles')
    ax.set_xlabel('Step')
    ax.set_ylabel('Free Energy (kcal/mol)')
    ax.legend()
    ax.grid(True)
    
    # Energy barriers
    ax = axs[0, 1]
    barriers = [results[method]['energy_barrier'] for method in methods]
    ax.bar(methods, barriers)
    ax.set_title('Energy Barriers')
    ax.set_xlabel('Method')
    ax.set_ylabel('Energy Barrier (kcal/mol)')
    for i, v in enumerate(barriers):
        ax.text(i, v + 0.1, f"{v:.2f}", ha='center')
    
    # Path lengths
    ax = axs[1, 0]
    path_lengths = [results[method]['path_length'] for method in methods]
    ax.bar(methods, path_lengths)
    ax.set_title('Path Lengths')
    ax.set_xlabel('Method')
    ax.set_ylabel('Number of Steps')
    for i, v in enumerate(path_lengths):
        ax.text(i, v + 0.5, str(v), ha='center')
    
    # Execution times
    ax = axs[1, 1]
    times = [results[method]['execution_time'] for method in methods]
    ax.bar(methods, times)
    ax.set_title('Execution Times')
    ax.set_xlabel('Method')
    ax.set_ylabel('Time (seconds)')
    for i, v in enumerate(times):
        ax.text(i, v + 0.1, f"{v:.2f}s", ha='center')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def transition_analysis(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Analyze transition types in different paths.
    
    Args:
        results: Results from compare_algorithms
        
    Returns:
        DataFrame with transition analysis
    """
    data = []
    
    for method, metrics in results.items():
        add_count = metrics.get('add_transitions', 0)
        remove_count = metrics.get('remove_transitions', 0)
        both_count = metrics.get('both_transitions', 0)
        total = add_count + remove_count + both_count
        
        # Calculate percentages
        add_pct = (add_count / total * 100) if total > 0 else 0
        remove_pct = (remove_count / total * 100) if total > 0 else 0
        both_pct = (both_count / total * 100) if total > 0 else 0
        
        data.append({
            'Method': method.capitalize(),
            'Add Transitions': add_count,
            'Remove Transitions': remove_count,
            'Both Transitions': both_count,
            'Total Transitions': total,
            'Add %': add_pct,
            'Remove %': remove_pct,
            'Both %': both_pct
        })
    
    return pd.DataFrame(data)


def energy_landscape_visualization(sequence: str, structures: List[str], 
                                  title="RNA Energy Landscape", figsize=(12, 8),
                                  point_labels=None, highlight_paths=None):
    """
    Visualize the energy landscape and paths between RNA structures.
    
    Args:
        sequence: RNA sequence
        structures: List of RNA structures in dot-bracket notation
        title: Plot title
        figsize: Figure size
        point_labels: Optional list of labels for points
        highlight_paths: Optional dictionary mapping path names to lists of structures
    """
    # Create ViennaRNA fold compound
    fc = RNA.fold_compound(sequence)
    
    # Calculate energies
    energies = [energy.turner_energy(fc, struct) for struct in structures]
    
    # Calculate pairwise distances
    n = len(structures)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = sum(1 for a, b in zip(structures[i], structures[j]) if a != b)
            distances[i, j] = d
            distances[j, i] = d
    
    # Dimensionality reduction
    if n > 2:
        # Use t-SNE for more than 2 points
        tsne = TSNE(n_components=2, random_state=42, perplexity=5) #metric='precomputed',
        points = tsne.fit_transform(distances)
    else:
        # For 2 or fewer points, just use a line
        points = np.array([[0, 0], [1, 0]])[:n]
    
    # Plot the landscape
    plt.figure(figsize=figsize)
    
    # Size by energy
    sizes = 100 + 20 * (max(energies) - np.array(energies))
    
    # Plot all points
    scatter = plt.scatter(points[:, 0], points[:, 1], c=energies, cmap='viridis_r',
               s=sizes, alpha=0.8, edgecolors='black')
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Free Energy (kcal/mol)')
    
    # Add labels if provided
    if point_labels:
        for i, label in enumerate(point_labels):
            if label:  # Only add non-empty labels
                plt.annotate(label, (points[i, 0], points[i, 1]),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=12, fontweight='bold')
    
    # Highlight paths
    if highlight_paths:
        for path_name, path in highlight_paths.items():
            # Find indices of structures in the path
            path_indices = [structures.index(struct) for struct in path if struct in structures]
            
            # Get points for the path
            path_points = points[path_indices]
            
            # Plot the path
            plt.plot(path_points[:, 0], path_points[:, 1], 'o-', 
                    linewidth=2, markersize=0, label=path_name.capitalize())
    
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def batch_analysis(sequences: List[str], start_structs: List[str], end_structs: List[str],
                  sequence_names=None, methods=['direct', 'indirect', 'evolutionary'],
                  **kwargs) -> pd.DataFrame:
    """
    Run analysis on multiple RNA sequences.
    
    Args:
        sequences: List of RNA sequences
        start_structs: List of starting structures
        end_structs: List of ending structures
        sequence_names: Optional list of names for the sequences
        methods: List of methods to compare
        **kwargs: Additional arguments for compare_algorithms
        
    Returns:
        DataFrame with results for all sequences and methods
    """
    if sequence_names is None:
        sequence_names = [f"RNA {i+1}" for i in range(len(sequences))]
    
    data = []
    
    for i, (seq, start, end, name) in enumerate(zip(sequences, start_structs, end_structs, sequence_names)):
        print(f"Analyzing {name}...")
        
        results = compare_algorithms(seq, start, end, methods=methods, **kwargs)
        
        for method, metrics in results.items():
            data.append({
                'Sequence': name,
                'Length': len(seq),
                'Method': method.capitalize(),
                'Energy Barrier': metrics['energy_barrier'],
                'Path Length': metrics['path_length'],
                'Execution Time (s)': metrics['execution_time'],
                'Initial Energy': metrics['initial_energy'],
                'Final Energy': metrics['final_energy'],
                'Average Energy': metrics['average_energy']
            })
    
    return pd.DataFrame(data)


def plot_sequence_comparison(batch_results: pd.DataFrame, metric='Energy Barrier', 
                           figsize=(12, 6), save_path=None):
    """
    Plot a comparison of a specific metric across sequences and methods.
    
    Args:
        batch_results: Results from batch_analysis
        metric: Metric to compare
        figsize: Figure size
        save_path: Path to save the figure (if provided)
    """
    plt.figure(figsize=figsize)
    
    # Create grouped bar chart
    sns.barplot(x='Sequence', y=metric, hue='Method', data=batch_results)
    
    plt.title(f'Comparison of {metric} Across RNA Sequences')
    plt.ylabel(metric)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(title='Method')
    
    if save_path:
        plt.savefig(save_path)
    
    plt.tight_layout()
    plt.show()


def correlation_analysis(batch_results: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze correlations between sequence properties and algorithm performance.
    
    Args:
        batch_results: Results from batch_analysis
        
    Returns:
        DataFrame with correlation analysis
    """
    # Create correlation matrix
    corr_metrics = ['Length', 'Energy Barrier', 'Path Length', 'Execution Time (s)',
                   'Initial Energy', 'Final Energy']
    
    # Prepare data by method
    methods = batch_results['Method'].unique()
    corr_dfs = {}
    
    for method in methods:
        method_data = batch_results[batch_results['Method'] == method][corr_metrics]
        corr_dfs[method] = method_data.corr()
    
    # Combine into a single DataFrame
    correlation_results = pd.concat(corr_dfs, names=['Method'])
    
    return correlation_results


def plot_correlation_matrix(correlation_results: pd.DataFrame, figsize=(12, 10), save_path=None):
    """
    Plot correlation matrices for each method.
    
    Args:
        correlation_results: Results from correlation_analysis
        figsize: Figure size
        save_path: Path to save the figure (if provided)
    """
    methods = correlation_results.index.levels[0]
    n_methods = len(methods)
    
    fig, axs = plt.subplots(1, n_methods, figsize=figsize)
    if n_methods == 1:
        axs = [axs]
    
    for i, method in enumerate(methods):
        corr_matrix = correlation_results.loc[method]
        
        # Plot correlation matrix
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                   center=0, fmt='.2f', ax=axs[i])
        
        axs[i].set_title(f'{method} Correlations')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def analyze_success_rates(sequence: str, start_struct: str, end_struct: str, 
                         num_trials=10, methods=['direct', 'indirect', 'evolutionary'],
                         energy_threshold=None, time_limit=None, **kwargs) -> pd.DataFrame:
    """
    Analyze the success rates of different algorithms based on custom criteria.
    
    Args:
        sequence: RNA sequence
        start_struct: Starting structure
        end_struct: Target structure
        num_trials: Number of trials for each method
        methods: List of methods to analyze
        energy_threshold: Maximum acceptable energy barrier (if None, no threshold)
        time_limit: Maximum acceptable execution time in seconds (if None, no limit)
        **kwargs: Additional arguments for compare_algorithms
        
    Returns:
        DataFrame with success rates
    """
    success_counts = {method: 0 for method in methods}
    average_metrics = {method: {'energy_barrier': 0, 'path_length': 0, 'execution_time': 0} 
                      for method in methods}
    
    for trial in range(num_trials):
        print(f"Trial {trial+1}/{num_trials}")
        
        results = compare_algorithms(sequence, start_struct, end_struct, methods=methods, **kwargs)
        
        for method, metrics in results.items():
            # Check if successful based on criteria
            is_successful = True
            
            if energy_threshold is not None and metrics['energy_barrier'] > energy_threshold:
                is_successful = False
            
            if time_limit is not None and metrics['execution_time'] > time_limit:
                is_successful = False
            
            if is_successful:
                success_counts[method] += 1
            
            # Accumulate metrics for averaging
            average_metrics[method]['energy_barrier'] += metrics['energy_barrier']
            average_metrics[method]['path_length'] += metrics['path_length']
            average_metrics[method]['execution_time'] += metrics['execution_time']
    
    # Calculate averages
    for method in methods:
        for metric in average_metrics[method]:
            average_metrics[method][metric] /= num_trials
    
    # Create DataFrame
    data = []
    for method in methods:
        success_rate = (success_counts[method] / num_trials) * 100
        data.append({
            'Method': method.capitalize(),
            'Success Count': success_counts[method],
            'Total Trials': num_trials,
            'Success Rate (%)': success_rate,
            'Avg Energy Barrier': average_metrics[method]['energy_barrier'],
            'Avg Path Length': average_metrics[method]['path_length'],
            'Avg Execution Time (s)': average_metrics[method]['execution_time']
        })
    
    return pd.DataFrame(data) 