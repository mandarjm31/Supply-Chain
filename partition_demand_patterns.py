import pandas as pd
import numpy as np
from pathlib import Path

class DemandPatternPartitioner:
    """
    Partitions time series data into demand pattern categories:
    - Smooth: Low variability, consistent demand
    - Intermittent: Many zero periods, low variability when demand occurs
    - Lumpy: Many zero periods, high variability in demand size
    - Erratic: Few zero periods, high variability in demand
    """
    
    def __init__(self, input_file, output_dir='demand_patterns'):
        """
        Initialize the partitioner.
        
        Args:
            input_file: Path to the CSV file
            output_dir: Directory to save partitioned datasets
        """
        self.input_file = input_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.df = None
        self.partitions = {
            'smooth': [],
            'intermittent': [],
            'lumpy': [],
            'erratic': []
        }
    
    def load_data(self):
        """Load the CSV file."""
        self.df = pd.read_csv(self.input_file)
        print(f"Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        print(f"Columns: {self.df.columns.tolist()}\n")
        return self.df
    
    def calculate_statistics(self, series):
        """
        Calculate key statistics for a time series.
        
        Returns:
            dict: Contains mean, std, cv (coefficient of variation), 
                  non_zero_count, zero_ratio
        """
        non_zero = series[series != 0]
        non_zero_count = len(non_zero)
        total_count = len(series)
        zero_ratio = 1 - (non_zero_count / total_count) if total_count > 0 else 1
        
        if len(non_zero) > 1:
            mean_val = non_zero.mean()
            std_val = non_zero.std()
            cv = std_val / mean_val if mean_val != 0 else 0  # Coefficient of Variation
        else:
            mean_val = non_zero.mean() if len(non_zero) > 0 else 0
            std_val = 0
            cv = 0
        
        return {
            'mean': mean_val,
            'std': std_val,
            'cv': cv,  # Coefficient of Variation
            'non_zero_count': non_zero_count,
            'zero_ratio': zero_ratio,
            'total_count': total_count
        }
    
    def classify_pattern(self, stats):
        """
        Classify a time series into demand patterns based on statistics.
        
        Classification Logic:
        - Smooth: Low CV (< 0.5) and few zero periods (zero_ratio < 0.3)
        - Intermittent: Many zero periods (zero_ratio >= 0.3) and low CV (< 0.5)
        - Lumpy: Many zero periods (zero_ratio >= 0.3) and high CV (>= 0.5)
        - Erratic: Few zero periods (zero_ratio < 0.3) and high CV (>= 0.5)
        
        Args:
            stats: Dictionary with statistical measures
            
        Returns:
            str: Pattern classification
        """
        cv_threshold = 0.5
        zero_ratio_threshold = 0.3
        
        cv = stats['cv']
        zero_ratio = stats['zero_ratio']
        
        # Many zero periods
        if zero_ratio >= zero_ratio_threshold:
            if cv >= cv_threshold:
                return 'lumpy'
            else:
                return 'intermittent'
        # Few zero periods
        else:
            if cv >= cv_threshold:
                return 'erratic'
            else:
                return 'smooth'
    
    def partition_data(self):
        """
        Partition the data by item and store into demand pattern categories.
        """
        if self.df is None:
            self.load_data()
        
        # Group by item_id (or the identifier column for individual series)
        # Assuming the data has store_id, item_id, and sell_price
        grouped = self.df.groupby(['store_id', 'item_id'])
        
        print(f"Processing {len(grouped)} unique item-store combinations...\n")
        
        for (store_id, item_id), group in grouped:
            # Get the price series
            series = group['sell_price'].values
            
            # Skip if too few data points
            if len(series) < 5:
                continue
            
            # Calculate statistics
            stats = self.calculate_statistics(series)
            
            # Classify pattern
            pattern = self.classify_pattern(stats)
            
            # Store the group data with classification info
            group_copy = group.copy()
            group_copy['pattern'] = pattern
            group_copy['stats'] = str(stats)
            
            self.partitions[pattern].append(group_copy)
            
            # Print classification info
            print(f"{store_id} - {item_id}: {pattern.upper()}")
            print(f"  Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}, CV: {stats['cv']:.3f}")
            print(f"  Zero Ratio: {stats['zero_ratio']:.3f}, Non-Zero Count: {stats['non_zero_count']}")
            print()
        
        return self.partitions
    
    def save_partitions(self):
        """Save each partition to a separate CSV file."""
        print("\n" + "="*80)
        print("SAVING PARTITIONS")
        print("="*80 + "\n")
        
        for pattern_name, data_list in self.partitions.items():
            if data_list:
                # Concatenate all groups for this pattern
                pattern_df = pd.concat(data_list, ignore_index=True)
                
                # Drop the temporary columns
                pattern_df = pattern_df.drop(columns=['pattern', 'stats'])
                
                # Save to CSV
                output_file = self.output_dir / f"{pattern_name}_demand.csv"
                pattern_df.to_csv(output_file, index=False)
                
                print(f"✓ {pattern_name.upper()} Demand:")
                print(f"  File: {output_file}")
                print(f"  Records: {len(pattern_df)}")
                print(f"  Unique Items: {pattern_df['item_id'].nunique()}")
                print()
            else:
                print(f"✗ {pattern_name.upper()} Demand: No data")
                print()
    
    def generate_summary_report(self):
        """Generate a summary report of the partitioning."""
        print("\n" + "="*80)
        print("SUMMARY REPORT")
        print("="*80 + "\n")
        
        total_records = sum(len(pd.concat(data)) if data else 0 
                           for data in self.partitions.values())
        
        for pattern_name, data_list in self.partitions.items():
            if data_list:
                pattern_df = pd.concat(data_list, ignore_index=True)
                record_count = len(pattern_df)
                percentage = (record_count / total_records * 100) if total_records > 0 else 0
                
                print(f"{pattern_name.upper()} DEMAND PATTERN")
                print(f"  Total Records: {record_count} ({percentage:.1f}%)")
                print(f"  Unique Store-Item Combinations: {len(data_list)}")
                print(f"  Date Range: {pattern_df['wm_yr_wk'].min()} to {pattern_df['wm_yr_wk'].max()}")
                print(f"  Price Range: ${pattern_df['sell_price'].min():.2f} - ${pattern_df['sell_price'].max():.2f}")
                print()
    
    def run(self):
        """Execute the complete partitioning pipeline."""
        print("="*80)
        print("DEMAND PATTERN PARTITIONER")
        print("="*80 + "\n")
        
        self.load_data()
        self.partition_data()
        self.save_partitions()
        self.generate_summary_report()
        
        print("="*80)
        print("PARTITIONING COMPLETE!")
        print("="*80)


def main():
    """Main execution function."""
    # Configuration
    input_file = 'timeseries_dataset_4_patterns_50KB.csv'  # Change to your file path
    output_directory = 'demand_patterns'
    
    # Create and run partitioner
    partitioner = DemandPatternPartitioner(input_file, output_directory)
    partitioner.run()


if __name__ == "__main__":
    main()