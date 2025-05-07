import pandas as pd
import os

def convert_excel_to_csv():
    """
    Convert each sheet of MAN-2025.xlsx into separate CSV files.
    Saves them in a new directory '2025 result/csv_files/'
    """
    # Input and output paths
    excel_path = './2025 result/MAN-2025.xlsx'
    output_dir = './2025 result/csv_files'
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Read all sheets from Excel file
        print(f"Reading Excel file: {excel_path}")
        excel_data = pd.read_excel(excel_path, sheet_name=None, engine='openpyxl')
        
        # Remove any non-data sheets
        for bad in ('Top-10', 'Signature'):
            excel_data.pop(bad, None)
        
        # Convert each sheet to CSV
        for sheet_name, df in excel_data.items():
            # Clean sheet name for file naming
            clean_name = sheet_name.replace(' ', '_').replace('/', '_')
            csv_path = os.path.join(output_dir, f"{clean_name}.csv")
            
            # Save to CSV
            df.to_csv(csv_path, index=False)
            print(f"Created: {csv_path}")
        
        print("\nConversion completed successfully!")
        print(f"CSV files are saved in: {output_dir}")
        
    except FileNotFoundError:
        print(f"Error: Could not find Excel file at {excel_path}")
    except Exception as e:
        print(f"Error during conversion: {e}")

if __name__ == "__main__":
    convert_excel_to_csv() 