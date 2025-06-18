#!/usr/bin/env python3
"""
Quick script to check EWoK dataset size for agentic domains
"""

import pandas as pd
from datasets import load_dataset

def check_ewok_dataset():
    """Check the size of EWoK dataset for agentic domains."""
    print("📥 Loading EWoK dataset...")
    
    try:
        # Load the dataset from the correct path
        dataset_path = "hf://datasets/ewok-core/ewok-core-1.0/data/test/ewok-core-1.0.parquet"
        df = pd.read_parquet(dataset_path)
        print("✅ Loaded from parquet file")
        
        print(f"✅ Total EWoK test samples: {len(df)}")
        
        # Show available domains
        domain_col = 'domain' if 'domain' in df.columns else 'Domain'
        available_domains = df[domain_col].unique()
        print(f"📊 Available domains: {sorted(available_domains)}")
        
        # Filter for agentic domains (try both column name variants)
        agentic_domains = ['agent-properties', 'social-interactions', 'social-properties']
        
        domain_col = 'domain' if 'domain' in df.columns else 'Domain'
        print(f"📋 Using domain column: '{domain_col}'")
        
        filtered_df = df[df[domain_col].isin(agentic_domains)].copy()
        
        print(f"\n🎯 Filtered for agentic domains: {len(filtered_df)} samples")
        
        # Show breakdown by domain
        print("\n📈 Domain breakdown:")
        for domain in agentic_domains:
            count = len(filtered_df[filtered_df[domain_col] == domain])
            percentage = (count / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
            print(f"  • {domain}: {count} samples ({percentage:.1f}%)")
        
        # Show sample structure
        if len(filtered_df) > 0:
            print(f"\n📝 Sample columns: {list(filtered_df.columns)}")
            print(f"📝 First sample domain: {filtered_df.iloc[0][domain_col]}")
            print(f"📝 Sample data example:")
            for col in ['Context1', 'Context2', 'Target1', 'Target2']:
                if col in filtered_df.columns:
                    print(f"     {col}: {filtered_df.iloc[0][col][:100]}...")
        
        return len(filtered_df)
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return 0

if __name__ == "__main__":
    total_agentic_samples = check_ewok_dataset()
    print(f"\n✅ Result: {total_agentic_samples} samples available for evaluation")