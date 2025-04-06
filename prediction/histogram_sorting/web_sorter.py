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
        self.current_index = 0
        
    def _load_reviews(self) -> pd.DataFrame:
        """Load existing reviews or create new DataFrame if none exists."""
        if self.csv_path.exists():
            return pd.read_csv(self.csv_path)
        return pd.DataFrame(columns=['file_path', 'significance', 'sub_category'])
    
    def _save_reviews(self):
        """Save current reviews to CSV."""
        self.reviews.to_csv(self.csv_path, index=False)
    
    def get_current_image(self):
        """Get the current image path relative to the directory."""
        if self.current_index < len(self.png_files):
            return str(self.png_files[self.current_index].relative_to(self.directory))
        return None
    
    def save_review(self, significance: str, sub_category: str = None):
        """Save a review for the current image."""
        if self.current_index >= len(self.png_files):
            return False
            
        current_file = str(self.png_files[self.current_index].relative_to(self.directory))
        
        # Update or add review
        if current_file in self.reviews['file_path'].values:
            self.reviews.loc[self.reviews['file_path'] == current_file, 
                           ['significance', 'sub_category']] = [significance, sub_category]
        else:
            new_review = pd.DataFrame({
                'file_path': [current_file],
                'significance': [significance],
                'sub_category': [sub_category]
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
    return jsonify({
        'image_path': current_image,
        'total': len(sorter.png_files),
        'current': sorter.current_index + 1
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
    return jsonify({'has_next': has_next})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Web-based histogram image sorter')
    parser.add_argument('--directory', type=str, help='Directory containing PNG files to review')
    args = parser.parse_args()
    
    directory = args.directory if args.directory else os.getcwd()
    sorter = ImageSorter(directory)
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create the HTML template
    with open('templates/index.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Histogram Sorter</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .image-container {
            text-align: center;
            margin: 20px 0;
        }
        .image-container img {
            max-width: 100%;
            max-height: 600px;
        }
        .controls {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin: 20px 0;
        }
        button {
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
        }
        .progress {
            text-align: center;
            margin: 10px 0;
        }
        .sub-category {
            display: none;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div class="progress">
        <span id="progress">0/0</span>
    </div>
    <div class="image-container">
        <img id="current-image" src="" alt="Current image">
    </div>
    <div class="controls">
        <button onclick="review('0')">Skip</button>
        <button onclick="review('1')">True Significant</button>
        <button onclick="review('2')">Not Significant</button>
        <button onclick="review('3')">False Significant</button>
        <button onclick="review('4')">Unknown</button>
        <button onclick="review('5')">Re-review</button>
    </div>
    <div class="sub-category" id="sub-category">
        <button onclick="submitWithSubCategory('1')">Left skew/significant</button>
        <button onclick="submitWithSubCategory('2')">Other</button>
        <button onclick="submitWithSubCategory('3')">Right skew/significant</button>
    </div>

    <script>
        let currentSignificance = null;
        
        function updateImage() {
            fetch('/image')
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('current-image').src = '';
                        document.getElementById('progress').textContent = 'Complete!';
                        return;
                    }
                    document.getElementById('current-image').src = '/images/' + data.image_path;
                    document.getElementById('progress').textContent = `${data.current}/${data.total}`;
                });
        }
        
        function review(significance) {
            currentSignificance = significance;
            if (significance === '1' || significance === '3') {
                document.getElementById('sub-category').style.display = 'block';
            } else {
                submitReview(significance);
            }
        }
        
        function submitWithSubCategory(subCategory) {
            submitReview(currentSignificance, subCategory);
            document.getElementById('sub-category').style.display = 'none';
        }
        
        function submitReview(significance, subCategory = null) {
            fetch('/review', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    significance: significance,
                    sub_category: subCategory
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.has_next) {
                    updateImage();
                } else {
                    document.getElementById('current-image').src = '';
                    document.getElementById('progress').textContent = 'Complete!';
                }
            });
        }
        
        // Load first image
        updateImage();
    </script>
</body>
</html>
        ''')
    
    app.run(debug=True) 