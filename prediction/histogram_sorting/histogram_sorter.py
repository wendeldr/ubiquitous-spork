import os
import sys
import argparse
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from typing import Optional, Tuple

class HistogramSorter:
    def __init__(self, directory: str, review_mode: bool = False):
        self.directory = Path(directory)
        self.review_mode = review_mode
        self.csv_path = self.directory / "histogram_reviews.csv"
        self.reviews = self._load_reviews()
        
    def _load_reviews(self) -> pd.DataFrame:
        """Load existing reviews or create new DataFrame if none exists."""
        if self.csv_path.exists():
            return pd.read_csv(self.csv_path)
        return pd.DataFrame(columns=['file_path', 'significance', 'sub_category'])
    
    def _save_reviews(self):
        """Save current reviews to CSV."""
        self.reviews.to_csv(self.csv_path, index=False)
    
    def _get_significance_from_input(self) -> Optional[Tuple[str, Optional[str]]]:
        """Get user input for significance and sub-category."""
        print("\nEnter significance (0-5):")
        print("0) Skip this file")
        print("1) True significant")
        print("2) Not significant")
        print("3) False significant")
        print("4) Unknown")
        print("5) Re-review")
        
        significance = input("> ")
        
        if significance not in ['0', '1', '2', '3', '4', '5']:
            print("Invalid input. Please try again.")
            return None
        
        if significance == '0':
            return '0', None
            
        if significance in ['1', '3']:
            print("\nEnter sub-category (1-3):")
            print("1) Left skew/significant")
            print("2) Other")
            print("3) Right skew/significant")
            
            sub_category = input("> ")
            
            if sub_category not in ['1', '2', '3']:
                print("Invalid input. Please try again.")
                return None
            
            return significance, sub_category
        
        return significance, None
    
    def _display_image(self, image_path: Path):
        """Display the image using matplotlib."""
        plt.figure(figsize=(10, 6))
        img = mpimg.imread(image_path)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Reviewing: {image_path.name}")
        plt.show()
    
    def process_directory(self):
        """Process all PNG files in the directory."""
        png_files = list(self.directory.rglob('*.png'))
        total_files = len(png_files)
        
        for idx, png_file in enumerate(png_files, 1):
            relative_path = png_file.relative_to(self.directory)
            
            # Skip if already reviewed and not in review mode
            if not self.review_mode and str(relative_path) in self.reviews['file_path'].values:
                print(f"Skipping already reviewed file: {relative_path}")
                continue
            
            print(f"\nProcessing file {idx}/{total_files}: {relative_path}")
            
            while True:
                self._display_image(png_file)
                result = self._get_significance_from_input()
                
                if result is not None:
                    significance, sub_category = result
                    
                    if significance == '0':
                        print("Skipping this file...")
                        break
                    
                    # Update or add review
                    if str(relative_path) in self.reviews['file_path'].values:
                        self.reviews.loc[self.reviews['file_path'] == str(relative_path), 
                                       ['significance', 'sub_category']] = [significance, sub_category]
                    else:
                        new_review = pd.DataFrame({
                            'file_path': [str(relative_path)],
                            'significance': [significance],
                            'sub_category': [sub_category]
                        })
                        self.reviews = pd.concat([self.reviews, new_review], ignore_index=True)
                    
                    self._save_reviews()
                    break

def main():
    parser = argparse.ArgumentParser(description='Review and classify histogram images.')
    parser.add_argument('--directory', type=str, help='Directory containing PNG files to review')
    parser.add_argument('--review', action='store_true', help='Review all files, including previously reviewed ones')
    
    args = parser.parse_args()
    
    directory = args.directory if args.directory else os.getcwd()
    sorter = HistogramSorter(directory, review_mode=args.review)
    sorter.process_directory()

if __name__ == "__main__":
    main()
