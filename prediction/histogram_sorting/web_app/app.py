from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from pathlib import Path
import pandas as pd
import argparse

app = Flask(__name__)

class ImageSorter:
    def __init__(self, directory: str):
        self.directory = Path(directory)
        self.csv_path = self.directory / "histogram_reviews.csv"
        self.reviews = self._load_reviews()
        self.png_files = list(self.directory.rglob('*.png'))
        self.current_index = self._get_starting_index()
        
        # Define significance and sub-category mappings
        self.significance_map = {
            '0': 'Skipped',
            '1': 'True significant',
            '2': 'Not significant',
            '3': 'False significant',
            '4': 'Unknown',
            '5': 'Re-review'
        }
        
        self.sub_category_map = {
            '1': 'Left skew/significant',
            '2': 'Other',
            '3': 'Right skew/significant'
        }
        
    def _load_reviews(self) -> pd.DataFrame:
        """Load existing reviews or create new DataFrame if none exists."""
        if self.csv_path.exists():
            return pd.read_csv(self.csv_path)
        return pd.DataFrame(columns=[
            'file_path', 
            'significance', 
            'significance_meaning',
            'sub_category',
            'sub_category_meaning'
        ])
    
    def _get_starting_index(self) -> int:
        """Get the index of the first unreviewed image."""
        if not self.reviews.empty:
            reviewed_files = set(self.reviews['file_path'].values)
            for i, png_file in enumerate(self.png_files):
                if str(png_file.relative_to(self.directory)) not in reviewed_files:
                    return i
            return len(self.png_files)  # All files reviewed
        return 0  # No reviews yet
    
    def _save_reviews(self):
        """Save current reviews to CSV."""
        self.reviews.to_csv(self.csv_path, index=False)
    
    def get_current_image(self):
        """Get the current image path relative to the directory."""
        if self.current_index < len(self.png_files):
            return str(self.png_files[self.current_index].relative_to(self.directory))
        return None
    
    def get_stats(self):
        """Get statistics about reviewed and remaining images."""
        total = len(self.png_files)
        reviewed = len(self.reviews)
        remaining = total - reviewed
        return {
            'total': total,
            'reviewed': reviewed,
            'remaining': remaining,
            'current': self.current_index + 1
        }
    
    def save_review(self, significance: str, sub_category: str = None):
        """Save a review for the current image."""
        if self.current_index >= len(self.png_files):
            return False
            
        current_file = str(self.png_files[self.current_index].relative_to(self.directory))
        
        # Get meanings
        significance_meaning = self.significance_map.get(significance, 'Unknown')
        sub_category_meaning = self.sub_category_map.get(sub_category, None)
        
        # Update or add review
        if current_file in self.reviews['file_path'].values:
            self.reviews.loc[self.reviews['file_path'] == current_file, 
                           ['significance', 'significance_meaning', 'sub_category', 'sub_category_meaning']] = [
                               significance, significance_meaning, sub_category, sub_category_meaning
                           ]
        else:
            new_review = pd.DataFrame({
                'file_path': [current_file],
                'significance': [significance],
                'significance_meaning': [significance_meaning],
                'sub_category': [sub_category],
                'sub_category_meaning': [sub_category_meaning]
            })
            self.reviews = pd.concat([self.reviews, new_review], ignore_index=True)
        
        self._save_reviews()
        return True
    
    def next_image(self):
        """Move to the next image."""
        self.current_index += 1
        return self.get_current_image() is not None

# Global sorter instance
sorter = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image')
def get_image():
    current_image = sorter.get_current_image()
    if current_image is None:
        return jsonify({'error': 'No more images'})
    stats = sorter.get_stats()
    return jsonify({
        'image_path': current_image,
        'stats': stats
    })

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(str(sorter.directory), filename)

@app.route('/review', methods=['POST'])
def review():
    data = request.json
    significance = data.get('significance')
    sub_category = data.get('sub_category')
    
    if significance is None:
        return jsonify({'error': 'Significance is required'}), 400
    
    if significance in ['1', '3'] and sub_category is None:
        return jsonify({'error': 'Sub-category is required for significance 1 or 3'}), 400
    
    success = sorter.save_review(significance, sub_category)
    if not success:
        return jsonify({'error': 'No current image'}), 400
    
    has_next = sorter.next_image()
    stats = sorter.get_stats()
    return jsonify({
        'has_next': has_next,
        'stats': stats
    })

def create_app(directory: str):
    global sorter
    sorter = ImageSorter(directory)
    return app 