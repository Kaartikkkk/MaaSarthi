document.addEventListener('DOMContentLoaded', () => {
    const searchInput = document.getElementById('jobSearchInput');
    const sortSelect = document.getElementById('jobSortSelect');
    const jobsList = document.querySelector('.jobs-list');
    const jobCards = Array.from(document.querySelectorAll('.job-card'));
    
    // Initialize saved jobs from localStorage
    let savedJobs = JSON.parse(localStorage.getItem('savedMaaSarthiJobs') || '[]');
    
    // Update UI for saved jobs
    jobCards.forEach(card => {
        const jobId = card.dataset.jobId;
        if (savedJobs.includes(jobId)) {
            card.classList.add('saved');
        }
        
        // Add click listener for save button
        const saveBtn = card.querySelector('.save-btn');
        if (saveBtn) {
            saveBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                toggleSaveJob(jobId, card);
            });
        }
    });

    // Live Search Filter
    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            const term = e.target.value.toLowerCase();
            jobCards.forEach(card => {
                const title = card.querySelector('.job-title').textContent.toLowerCase();
                const company = card.querySelector('.company-name').textContent.toLowerCase();
                if (title.includes(term) || company.includes(term)) {
                    card.style.display = 'block';
                } else {
                    card.style.display = 'none';
                }
            });
        });
    }

    // Dynamic Sorting
    if (sortSelect) {
        sortSelect.addEventListener('change', (e) => {
            const sortBy = e.target.value;
            let sortedCards;
            
            if (sortBy === 'salary') {
                sortedCards = jobCards.sort((a, b) => {
                    const salA = parseSalary(a.querySelector('.salary-value').textContent);
                    const salB = parseSalary(b.querySelector('.salary-value').textContent);
                    return salB - salA;
                });
            } else if (sortBy === 'match') {
                sortedCards = jobCards.sort((a, b) => {
                    const matchA = parseInt(a.dataset.match || '0');
                    const matchB = parseInt(b.dataset.match || '0');
                    return matchB - matchA;
                });
            } else {
                // Default order (original)
                sortedCards = jobCards.sort((a, b) => a.dataset.index - b.dataset.index);
            }
            
            // Re-append sorted cards
            sortedCards.forEach(card => jobsList.appendChild(card));
        });
    }

    function toggleSaveJob(id, card) {
        if (savedJobs.includes(id)) {
            savedJobs = savedJobs.filter(itemId => itemId !== id);
            card.classList.remove('saved');
        } else {
            savedJobs.push(id);
            card.classList.add('saved');
            showToast('Job saved to your profile!');
        }
        localStorage.setItem('savedMaaSarthiJobs', JSON.stringify(savedJobs));
    }

    function parseSalary(str) {
        // Simple parser for "₹15,000 - ₹20,000" or generic strings
        const match = str.replace(/,/g, '').match(/\d+/);
        return match ? parseInt(match[0]) : 0;
    }

    function showToast(msg) {
        const toast = document.createElement('div');
        toast.className = 'glass-toast';
        toast.textContent = msg;
        document.body.appendChild(toast);
        
        setTimeout(() => toast.classList.add('show'), 100);
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }
});
