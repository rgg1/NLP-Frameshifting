import pandas as pd
import os

def read_file_with_encoding(file_path: str, columns: list, encodings=['utf-8', 'iso-8859-1', 'cp1252', 'latin1']) -> pd.DataFrame:
    """
    Try to read file with different encodings.
    
    Args:
        file_path: Path to the file
        columns: Column names for the DataFrame
        encodings: List of encodings to try
        
    Returns:
        DataFrame with the file contents
    """
    for encoding in encodings:
        try:
            return pd.read_csv(
                file_path,
                sep='|',
                names=columns,
                encoding=encoding,
                on_bad_lines='skip',
                quoting=3,
                dtype=str
            )
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error with encoding {encoding}: {str(e)}")
            continue
    
    raise ValueError(f"Could not read file {file_path} with any of the attempted encodings")

def trim_file(input_file: str, keep_fraction: float = 0.6) -> None:
    """
    Trim a file to keep approximately the first portion of entries.
    Saves to a new file with '_trimmed' suffix.
    
    Args:
        input_file: Path to the speech or description file
        keep_fraction: Fraction of entries to keep (default 0.6 for 3/5)
    """
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return
    
    print(f"\nProcessing: {input_file}")
    
    # Get original file size
    original_size = os.path.getsize(input_file) / (1024 * 1024)  # MB
    print(f"Original size: {original_size:.2f} MB")
    
    # Create output filename with _trimmed suffix
    base, ext = os.path.splitext(input_file)
    output_file = f"{base}_trimmed{ext}"
    
    # Determine if it's a speech file or description file
    is_speech = 'speeches' in input_file
    
    # Set column names based on file type
    if is_speech:
        columns = ['speech_id', 'speech']
    else:  # description file
        columns = [
            'speech_id', 'chamber', 'date', 'file', 'line_start', 'line_end',
            'number_within_file', 'speaker', 'first_name', 'last_name', 
            'state', 'gender', 'word_count', 'char_count'
        ]
    
    try:
        # Read the file with appropriate encoding
        print("Reading file...")
        df = read_file_with_encoding(input_file, columns)
        
        # Calculate how many entries to keep
        n_keep = int(len(df) * keep_fraction)
        print(f"Keeping {n_keep} out of {len(df)} entries (first 3/5)")
        
        # Keep the first n_keep entries
        df_trimmed = df.iloc[:n_keep]
        
        # Save trimmed data to new file
        print(f"Saving trimmed file to: {output_file}")
        df_trimmed.to_csv(
            output_file,
            sep='|',
            index=False,
            header=False,
            encoding='utf-8',
            quoting=3
        )
        
        # Get new file size
        new_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        
        # Print statistics
        print(f"New size: {new_size:.2f} MB")
        print(f"Reduction: {(original_size - new_size) / original_size * 100:.1f}%")
        print("Success!")
        
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")

def main():
    """Process all speech and description files."""
    print("Starting data shortening process...")
    
    # Get current directory (where the script is located)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # List of files to process
    files = [
        'speeches_113.txt',
        'speeches_114.txt',
    ]
    
    # Process each file
    for file in files:
        file_path = os.path.join(current_dir, file)
        trim_file(file_path)
    
    print("\nData shortening complete!")
    print("New files have been created with '_trimmed' suffix.")
    print("Original files have been preserved.")

if __name__ == "__main__":
    main()
