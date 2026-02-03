let currentCardIndex = 0;
let currentFilter = 'all';
let filteredCards = [...flashcardsData];
let studiedCards = new Set();

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadCard();
    updateStats();
    setupKeyboardNavigation();
});

function loadCard() {
    const card = filteredCards[currentCardIndex];
    const flashcard = document.getElementById('flashcard');
    
    // Remove flipped class
    flashcard.classList.remove('flipped');
    
    // Update content
    document.getElementById('cardNumber').textContent = card.id;
    document.getElementById('question').textContent = card.question;
    document.getElementById('answer').textContent = card.answer;
    document.getElementById('categoryName').textContent = card.categoryName;
    
    // Update progress
    document.getElementById('currentCard').textContent = currentCardIndex + 1;
    document.getElementById('totalCards').textContent = filteredCards.length;
    
    const progressPercent = ((currentCardIndex + 1) / filteredCards.length) * 100;
    document.getElementById('progressFill').style.width = progressPercent + '%';
    
    // Update button states
    document.getElementById('prevBtn').disabled = currentCardIndex === 0;
    document.getElementById('nextBtn').disabled = currentCardIndex === filteredCards.length - 1;
    
    // Mark as studied
    studiedCards.add(card.id);
    updateStats();
}

function flipCard() {
    const flashcard = document.getElementById('flashcard');
    flashcard.classList.toggle('flipped');
}

function nextCard() {
    if (currentCardIndex < filteredCards.length - 1) {
        currentCardIndex++;
        loadCard();
    }
}

function previousCard() {
    if (currentCardIndex > 0) {
        currentCardIndex--;
        loadCard();
    }
}

function filterCategory(category) {
    currentFilter = category;
    
    // Update active button
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');
    
    // Filter cards
    if (category === 'all') {
        filteredCards = [...flashcardsData];
    } else {
        filteredCards = flashcardsData.filter(card => card.category === category);
    }
    
    // Reset to first card
    currentCardIndex = 0;
    loadCard();
}

function updateStats() {
    document.getElementById('studiedCount').textContent = studiedCards.size;
    document.getElementById('remainingCount').textContent = flashcardsData.length - studiedCards.size;
}

function setupKeyboardNavigation() {
    document.addEventListener('keydown', (e) => {
        switch(e.key) {
            case 'ArrowLeft':
                previousCard();
                break;
            case 'ArrowRight':
                nextCard();
                break;
            case ' ':
                e.preventDefault();
                flipCard();
                break;
        }
    });
}

// Click on card to flip
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('flashcard').addEventListener('click', flipCard);
});
