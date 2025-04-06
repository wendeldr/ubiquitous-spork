import os
import argparse
from app import create_app

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Web-based histogram image sorter')
    parser.add_argument('--directory', type=str, help='Directory containing PNG files to review')
    args = parser.parse_args()
    
    directory = args.directory if args.directory else os.getcwd()
    app = create_app(directory)
    app.run(debug=True) 