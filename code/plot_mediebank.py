import pandas as pd
import matplotlib.pyplot as plt

def plot_hits(start_year, end_year):
    # Load the CSV file into a DataFrame
    file_path = f'./data/scrape_mediestream_{start_year}_{end_year}.csv'
    df = pd.read_csv(file_path)

    # Ensure that the 'year' column is integer and 'hits' column is numeric
    df['year'] = df['year'].astype(int)
    df['hits'] = pd.to_numeric(df['hits'], errors='coerce')

    # Plotting the data
    plt.figure(figsize=(10, 6))
    plt.plot(df['year'], df['hits'], marker='o', color='b', linestyle='-', linewidth=2, markersize=6)
    plt.title(f"Hits per Year: {start_year} to {end_year}", fontsize=14)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Number of Hits", fontsize=12)
    plt.grid(True)

    # theme
    plt.tight_layout()
    
    # export 
    plt.savefig(f'./output/medie_stream_{start_year}_{end_year}.png')
    plt.show()

# Example usage:
if __name__ == "__main__":
    start_year = 1850
    end_year = 2010
    plot_hits(start_year, end_year)
