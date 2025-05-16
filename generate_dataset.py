import numpy as np
import pandas as pd

def generate_gender_dataset(n_samples=100, random_seed=42):
    """
    Generate a synthetic dataset of height, weight, and gender.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate (will be evenly split between males and females)
    random_seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns: height, weight, gender (0=male, 1=female)
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Ensure even split between genders
    n_males = n_samples // 2
    n_females = n_samples - n_males
    
    # Generate male samples (taller and heavier on average)
    male_heights = np.random.normal(178, 8, n_males)  # Mean 178cm, std 8cm
    male_heights = np.clip(male_heights, 165, 200)    # Clip to realistic range
    male_heights = np.round(male_heights).astype(int)  # Convert to integers
    
    male_weights = np.random.normal(80, 10, n_males)  # Mean 80kg, std 10kg
    male_weights = np.clip(male_weights, 60, 100)     # Clip to realistic range
    male_weights = np.round(male_weights).astype(int)  # Convert to integers
    
    # Generate female samples (shorter and lighter on average)
    female_heights = np.random.normal(165, 7, n_females)  # Mean 165cm, std 7cm
    female_heights = np.clip(female_heights, 150, 185)    # Clip to realistic range
    female_heights = np.round(female_heights).astype(int)  # Convert to integers
    
    female_weights = np.random.normal(62, 9, n_females)   # Mean 62kg, std 9kg
    female_weights = np.clip(female_weights, 45, 85)      # Clip to realistic range
    female_weights = np.round(female_weights).astype(int)  # Convert to integers
    
    # Combine data
    heights = np.concatenate([male_heights, female_heights])
    weights = np.concatenate([male_weights, female_weights])
    genders = np.concatenate([np.zeros(n_males), np.ones(n_females)])
    
    # Create DataFrame
    df = pd.DataFrame({
        'height': heights,
        'weight': weights,
        'gender': genders
    })
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    return df

def main():
    # Generate dataset with 120 samples (60 males, 60 females)
    df = generate_gender_dataset(n_samples=120, random_seed=42)
    
    # Display dataset statistics
    print(f"Dataset shape: {df.shape}")
    print("\nSample of the dataset:")
    print(df.head(10))
    
    print("\nGender distribution:")
    print(df['gender'].value_counts())
    
    print("\nStatistics by gender:")
    print(df.groupby('gender').agg({'height': ['mean', 'min', 'max'], 
                                    'weight': ['mean', 'min', 'max']}))
    
    # Save to CSV
    csv_path = 'gender_data.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nDataset saved to {csv_path}")

if __name__ == "__main__":
    main() 