import pandas as pd
from pathlib import Path


def read_data(path_name):
    df = pd.read_excel(path_name)  # Use read_excel() instead of read_csv()

    if 'cnda_session_label' in df.columns:
        print("The column exists in the Excel file")
        print(df['cnda_session_label'].unique())
        print("Number of sessions: ", df['cnda_session_label'].nunique())
    else:
        print("The column does not exist in the Excel file")


def main():
    file_path = Path(r"C:\Users\aaron.l\Documents\db_20241213.xlsx")
    read_data(file_path)


if __name__ == "__main__":
    main()
