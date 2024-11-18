import os
import sys

import pandas as pd
import argparse


def check_path(path, file_usage='r'):
    """Check if a path exists and is accessible."""
    if not os.path.exists(path):
        print("Error! The path '{}' does not exist.".format(path))
        sys.exit(1)
    return True

def get_parser():
    """gathering parameters"""
    parser = argparse.ArgumentParser(description="Path to model_dir")
    
    # Define the argument
    parser.add_argument(
        '-p', '--path',  # Short and long version of the argument
        help="Path to parameters from the experiments params.json file in model_dir",
        required=True,    # Make this argument mandatory
    )

    parser.add_argument(
        '-s','--store',
        help="Path to store data",
        required=True,    # Make this argument mandatory
    )

    # Parse the arguments
    args = parser.parse_args()
    
    # Check the validity of the provided path
    check_path(args.path)
    
    return args

def get_data(path:str):

    print("Reading data...")

    df = pd.read_csv(path)

    df['Date'] = pd.to_datetime(df['date_created'], errors='coerce')
    df['Month'] = df['Date'].dt.month
    df['Year']  = df['Date'].dt.year

    monthly_groups = df.groupby(['Year','Month'])
    return monthly_groups

def store_data(path:str,data):
    print("Storing data...")
    output_dir=path

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for (y,m), group in data:
        f_name = "{}_{:02d}.csv".format(int(y),int(m))
        f_path = os.path.join(output_dir,f_name)

        group.to_csv(f_path, index=False)

def main():
    args = get_parser()
    monthly_groups = get_data(args.path)
    store_data(args.store,monthly_groups)

if __name__ == "__main__":
    main()