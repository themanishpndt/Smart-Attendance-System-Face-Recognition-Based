// Attendance Dashboard JavaScript
document.addEventListener('DOMContentLoaded', function() {
    initializeDateInput();
    initializeTableSearch();
    initializeTooltips();
    animateStats();
});

/**
 * Initialize date input with today's date if not set
 */
function initializeDateInput() {
    const dateInput = document.querySelector('input[name="date"]');
    if (dateInput && !dateInput.value) {
        const today = new Date();
        const dd = String(today.getDate()).padStart(2, '0');
        const mm = String(today.getMonth() + 1).padStart(2, '0');
        const yyyy = today.getFullYear();
        dateInput.value = `${yyyy}-${mm}-${dd}`;
    }
}

/**
 * Initialize client-side table search functionality
 */
function initializeTableSearch() {
    const searchInput = document.querySelector('input[name="name"]');
    if (searchInput) {
        // Add debounce to search
        let searchTimeout;
        searchInput.addEventListener('input', function() {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {
                filterTableRows(this.value.toLowerCase());
            }, 300);
        });
    }
}

/**
 * Filter table rows based on search term
 */
function filterTableRows(searchTerm) {
    const table = document.getElementById('attendanceTable');
    if (!table) return;
    
    const rows = table.querySelectorAll('tbody tr');
    let visibleCount = 0;
    
    rows.forEach(row => {
        const cells = row.querySelectorAll('td');
        let shouldShow = false;
        
        cells.forEach(cell => {
            if (cell.textContent.toLowerCase().includes(searchTerm)) {
                shouldShow = true;
            }
        });
        
        if (shouldShow) {
            row.style.display = '';
            visibleCount++;
        } else {
            row.style.display = 'none';
        }
    });
    
    // Update visible count in table title
    updateVisibleCount(visibleCount);
}

/**
 * Update the visible count in table title
 */
function updateVisibleCount(count) {
    const countSpan = document.querySelector('.table-title span');
    if (countSpan) {
        countSpan.textContent = `(${count} records)`;
    }
}

/**
 * Export table data to CSV
 */
function exportToCSV() {
    const table = document.getElementById('attendanceTable');
    if (!table) {
        alert('No table found to export');
        return;
    }
    
    const rows = table.querySelectorAll('tr');
    const csv = [];
    
    // Process all rows (including header)
    rows.forEach(row => {
        // Only include visible rows
        if (row.style.display !== 'none') {
            const cols = row.querySelectorAll('td, th');
            const csvRow = [];
            
            cols.forEach(col => {
                // Get text content and clean it
                let text = col.textContent.trim();
                // Remove extra whitespace
                text = text.replace(/\s+/g, ' ');
                // Escape quotes
                text = text.replace(/"/g, '""');
                csvRow.push(`"${text}"`);
            });
            
            csv.push(csvRow.join(','));
        }
    });
    
    // Create CSV content
    const csvContent = csv.join('\n');
    
    // Create blob and download
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    
    // Get current date for filename
    const filterDate = document.querySelector('input[name="date"]')?.value || 'all';
    const fileName = `attendance_${filterDate}_${new Date().getTime()}.csv`;
    
    // Handle different browsers
    if (navigator.msSaveBlob) {
        // IE 10+
        navigator.msSaveBlob(blob, fileName);
    } else {
        link.href = URL.createObjectURL(blob);
        link.download = fileName;
        link.style.display = 'none';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
    
    // Show success message
    showNotification('CSV exported successfully!', 'success');
}

/**
 * Show notification message
 */
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 25px;
        background: ${type === 'success' ? '#43e97b' : '#667eea'};
        color: white;
        border-radius: 10px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.2);
        z-index: 9999;
        animation: slideIn 0.3s ease-out;
    `;
    
    document.body.appendChild(notification);
    
    // Remove after 3 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-out';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 3000);
}

/**
 * Initialize tooltips for badges
 */
function initializeTooltips() {
    const badges = document.querySelectorAll('.id-badge, .dept-badge');
    badges.forEach(badge => {
        badge.title = badge.textContent;
    });
}

/**
 * Animate statistics cards on load
 */
function animateStats() {
    const statValues = document.querySelectorAll('.stat-value');
    statValues.forEach(stat => {
        const finalValue = parseInt(stat.textContent);
        if (isNaN(finalValue)) return;
        
        let currentValue = 0;
        const increment = Math.ceil(finalValue / 30);
        const duration = 1000;
        const stepTime = duration / (finalValue / increment);
        
        const timer = setInterval(() => {
            currentValue += increment;
            if (currentValue >= finalValue) {
                stat.textContent = finalValue;
                clearInterval(timer);
            } else {
                stat.textContent = currentValue;
            }
        }, stepTime);
    });
}

/**
 * Print attendance report
 */
function printReport() {
    window.print();
}

/**
 * Reset all filters
 */
function resetFilters() {
    const form = document.getElementById('filterForm');
    if (form) {
        form.reset();
        // Set date to today
        const dateInput = form.querySelector('input[name="date"]');
        if (dateInput) {
            const today = new Date();
            const dd = String(today.getDate()).padStart(2, '0');
            const mm = String(today.getMonth() + 1).padStart(2, '0');
            const yyyy = today.getFullYear();
            dateInput.value = `${yyyy}-${mm}-${dd}`;
        }
        form.submit();
    }
}

/**
 * Highlight today's records
 */
function highlightTodayRecords() {
    const today = new Date().toLocaleDateString('en-GB', {
        day: '2-digit',
        month: '2-digit',
        year: 'numeric'
    }).replace(/\//g, '-');
    
    const table = document.getElementById('attendanceTable');
    if (!table) return;
    
    const rows = table.querySelectorAll('tbody tr');
    rows.forEach(row => {
        const dateCell = row.querySelector('td:nth-child(5)'); // Date column
        if (dateCell && dateCell.textContent.trim() === today) {
            row.style.backgroundColor = 'rgba(67, 233, 123, 0.1)';
        }
    });
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Make functions globally available
window.exportToCSV = exportToCSV;
window.printReport = printReport;
window.resetFilters = resetFilters;
