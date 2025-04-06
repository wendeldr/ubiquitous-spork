let currentSignificance = null;
let isCompleted = false;

// Add keyboard event listener
document.addEventListener('keydown', function(event) {
    // Ignore keyboard events if completed
    if (isCompleted) return;
    
    // If we're in sub-category mode (1 or 3 was selected)
    if (currentSignificance && (currentSignificance === '1' || currentSignificance === '3')) {
        // Only handle 1-3 for sub-category selection
        if (event.key >= '1' && event.key <= '3') {
            submitWithSubCategory(event.key);
        }
    } else {
        // Handle main category selection (0-5)
        if (event.key >= '0' && event.key <= '5') {
            review(event.key);
        }
    }
});

function updateImage() {
    fetch('/image')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                handleCompletion();
                return;
            }
            document.getElementById('current-image').src = '/images/' + data.image_path;
            updateStats(data.stats);
        });
}

function updateStats(stats) {
    document.getElementById('current').textContent = stats.current;
    document.getElementById('reviewed').textContent = stats.reviewed;
    document.getElementById('remaining').textContent = stats.remaining;
    document.getElementById('total').textContent = stats.total;
}

function handleCompletion() {
    isCompleted = true;
    document.getElementById('current-image').src = '';
    document.body.classList.add('completed');
    document.getElementById('completion-message').style.display = 'block';
    
    // Disable all buttons
    document.querySelectorAll('button').forEach(button => {
        button.disabled = true;
    });
}

function showButtonFeedback(button) {
    // Remove selected class from all buttons
    document.querySelectorAll('button').forEach(btn => btn.classList.remove('selected'));
    
    // Add selected class to the clicked button
    button.classList.add('selected');
    
    // Remove the selected class after a short delay
    setTimeout(() => {
        button.classList.remove('selected');
    }, 250);
}

function review(significance) {
    if (isCompleted) return;
    
    currentSignificance = significance;
    const button = document.querySelector(`button[onclick="review('${significance}')"]`);
    showButtonFeedback(button);
    
    if (significance === '1' || significance === '3') {
        document.getElementById('sub-category').style.display = 'block';
    } else {
        document.getElementById('sub-category').style.display = 'none';
        submitReview(significance);
    }
}

function submitWithSubCategory(subCategory) {
    if (isCompleted) return;
    
    const button = document.querySelector(`button[onclick="submitWithSubCategory('${subCategory}')"]`);
    showButtonFeedback(button);
    
    // Store the current significance before resetting it
    const significance = currentSignificance;
    
    // Wait for the feedback animation to complete before hiding sub-category and submitting
    setTimeout(() => {
        document.getElementById('sub-category').style.display = 'none';
        currentSignificance = null;
        submitReview(significance, subCategory);
    }, 250);
}

function submitReview(significance, subCategory = null) {
    if (isCompleted) return;
    
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
        updateStats(data.stats);
        if (data.has_next) {
            updateImage();
        } else {
            handleCompletion();
        }
    });
}

// Load first image
updateImage(); 