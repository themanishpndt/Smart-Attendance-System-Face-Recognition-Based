// ===================================
//  Smart Attendance System - Main JavaScript
//  Enhanced Interactivity & User Experience
// ===================================

// ===== Theme Management =====
class ThemeManager {
    constructor() {
        this.theme = localStorage.getItem('theme') || 'light';
        this.applyTheme();
    }

    applyTheme() {
        document.documentElement.setAttribute('data-theme', this.theme);
        this.updateToggleButton();
    }

    toggle() {
        this.theme = this.theme === 'light' ? 'dark' : 'light';
        localStorage.setItem('theme', this.theme);
        this.applyTheme();
        this.animateThemeTransition();
    }

    updateToggleButton() {
        const toggleBtn = document.getElementById('theme-toggle');
        if (toggleBtn) {
            const icon = toggleBtn.querySelector('i');
            if (icon) {
                icon.className = this.theme === 'light' ? 'fas fa-moon' : 'fas fa-sun';
            }
        }
    }

    animateThemeTransition() {
        document.body.style.transition = 'background 0.3s ease, color 0.3s ease';
        setTimeout(() => {
            document.body.style.transition = '';
        }, 300);
    }
}

// ===== Notification System =====
class NotificationSystem {
    constructor() {
        this.container = this.createContainer();
    }

    createContainer() {
        let container = document.getElementById('notification-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'notification-container';
            container.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 9999;
                max-width: 400px;
            `;
            document.body.appendChild(container);
        }
        return container;
    }

    show(message, type = 'info', duration = 5000) {
        const notification = document.createElement('div');
        notification.className = `alert alert-${type} fade-in`;
        notification.style.cssText = `
            margin-bottom: 10px;
            animation: slideInRight 0.5s ease;
        `;
        
        const icon = this.getIcon(type);
        notification.innerHTML = `
            <div style="display: flex; align-items: center; gap: 10px;">
                <i class="${icon}" style="font-size: 1.5rem;"></i>
                <div style="flex: 1;">${message}</div>
                <button onclick="this.parentElement.parentElement.remove()" 
                        style="background: none; border: none; cursor: pointer; font-size: 1.2rem; color: inherit;">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;

        this.container.appendChild(notification);

        if (duration > 0) {
            setTimeout(() => {
                notification.style.animation = 'slideOutRight 0.5s ease';
                setTimeout(() => notification.remove(), 500);
            }, duration);
        }

        return notification;
    }

    getIcon(type) {
        const icons = {
            success: 'fas fa-check-circle',
            danger: 'fas fa-exclamation-circle',
            warning: 'fas fa-exclamation-triangle',
            info: 'fas fa-info-circle'
        };
        return icons[type] || icons.info;
    }
}

// ===== Scroll Animation Observer =====
class ScrollAnimator {
    constructor() {
        this.observer = new IntersectionObserver(
            (entries) => this.handleIntersection(entries),
            { threshold: 0.1, rootMargin: '50px' }
        );
        this.init();
    }

    init() {
        document.querySelectorAll('.card, .stat-card, .animate-on-scroll').forEach(el => {
            el.style.opacity = '0';
            el.style.transform = 'translateY(30px)';
            this.observer.observe(el);
        });
    }

    handleIntersection(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.transition = 'all 0.6s ease';
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
                this.observer.unobserve(entry.target);
            }
        });
    }
}

// ===== Form Validation =====
class FormValidator {
    constructor(formId) {
        this.form = document.getElementById(formId);
        if (this.form) {
            this.init();
        }
    }

    init() {
        this.form.addEventListener('submit', (e) => this.validate(e));
        
        // Real-time validation
        this.form.querySelectorAll('input, textarea, select').forEach(field => {
            field.addEventListener('blur', () => this.validateField(field));
            field.addEventListener('input', () => this.clearError(field));
        });
    }

    validate(e) {
        let isValid = true;
        const fields = this.form.querySelectorAll('input[required], textarea[required], select[required]');
        
        fields.forEach(field => {
            if (!this.validateField(field)) {
                isValid = false;
            }
        });

        if (!isValid) {
            e.preventDefault();
            this.showError('Please fill in all required fields correctly.');
        }

        return isValid;
    }

    validateField(field) {
        const value = field.value.trim();
        let isValid = true;
        let errorMessage = '';

        // Required check
        if (field.hasAttribute('required') && !value) {
            isValid = false;
            errorMessage = 'This field is required';
        }
        // Email validation
        else if (field.type === 'email' && value && !this.isValidEmail(value)) {
            isValid = false;
            errorMessage = 'Please enter a valid email address';
        }
        // Min length
        else if (field.hasAttribute('minlength') && value.length < parseInt(field.getAttribute('minlength'))) {
            isValid = false;
            errorMessage = `Minimum ${field.getAttribute('minlength')} characters required`;
        }
        // Max length
        else if (field.hasAttribute('maxlength') && value.length > parseInt(field.getAttribute('maxlength'))) {
            isValid = false;
            errorMessage = `Maximum ${field.getAttribute('maxlength')} characters allowed`;
        }

        if (!isValid) {
            this.showFieldError(field, errorMessage);
        } else {
            this.clearError(field);
        }

        return isValid;
    }

    isValidEmail(email) {
        return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
    }

    showFieldError(field, message) {
        this.clearError(field);
        field.classList.add('is-invalid');
        const errorDiv = document.createElement('div');
        errorDiv.className = 'invalid-feedback';
        errorDiv.style.display = 'block';
        errorDiv.textContent = message;
        field.parentNode.appendChild(errorDiv);
    }

    clearError(field) {
        field.classList.remove('is-invalid');
        const error = field.parentNode.querySelector('.invalid-feedback');
        if (error) {
            error.remove();
        }
    }

    showError(message) {
        window.notificationSystem.show(message, 'danger');
    }
}

// ===== Data Table Enhancement =====
class DataTableEnhancer {
    constructor(tableId) {
        this.table = document.getElementById(tableId);
        if (this.table) {
            this.init();
        }
    }

    init() {
        this.addSearchBox();
        this.addSortingCapability();
        this.addPagination();
        this.makeResponsive();
    }

    addSearchBox() {
        const searchBox = document.createElement('div');
        searchBox.className = 'mb-3';
        searchBox.innerHTML = `
            <input type="text" class="form-control" placeholder="🔍 Search table..." 
                   id="table-search" style="max-width: 300px;">
        `;
        this.table.parentNode.insertBefore(searchBox, this.table);

        document.getElementById('table-search').addEventListener('input', (e) => {
            this.filterTable(e.target.value);
        });
    }

    filterTable(searchTerm) {
        const rows = this.table.querySelectorAll('tbody tr');
        const term = searchTerm.toLowerCase();

        rows.forEach(row => {
            const text = row.textContent.toLowerCase();
            row.style.display = text.includes(term) ? '' : 'none';
        });
    }

    addSortingCapability() {
        const headers = this.table.querySelectorAll('thead th');
        headers.forEach((header, index) => {
            header.style.cursor = 'pointer';
            header.innerHTML += ' <i class="fas fa-sort" style="font-size: 0.8rem; opacity: 0.5;"></i>';
            
            header.addEventListener('click', () => this.sortTable(index));
        });
    }

    sortTable(columnIndex) {
        const tbody = this.table.querySelector('tbody');
        const rows = Array.from(tbody.querySelectorAll('tr'));
        const isAscending = this.table.getAttribute('data-sort-order') !== 'asc';

        rows.sort((a, b) => {
            const aText = a.cells[columnIndex].textContent.trim();
            const bText = b.cells[columnIndex].textContent.trim();
            
            if (!isNaN(aText) && !isNaN(bText)) {
                return isAscending ? aText - bText : bText - aText;
            }
            
            return isAscending ? aText.localeCompare(bText) : bText.localeCompare(aText);
        });

        rows.forEach(row => tbody.appendChild(row));
        this.table.setAttribute('data-sort-order', isAscending ? 'asc' : 'desc');
    }

    addPagination() {
        // Simple pagination for large tables
        const rows = this.table.querySelectorAll('tbody tr');
        if (rows.length > 10) {
            // Implement pagination if needed
        }
    }

    makeResponsive() {
        const wrapper = document.createElement('div');
        wrapper.className = 'table-responsive';
        this.table.parentNode.insertBefore(wrapper, this.table);
        wrapper.appendChild(this.table);
    }
}

// ===== Smooth Scroll =====
function smoothScrollTo(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

// ===== Loading Overlay =====
class LoadingOverlay {
    constructor() {
        this.overlay = this.createOverlay();
    }

    createOverlay() {
        const overlay = document.createElement('div');
        overlay.id = 'loading-overlay';
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.7);
            display: none;
            justify-content: center;
            align-items: center;
            z-index: 99999;
        `;
        overlay.innerHTML = `
            <div style="text-align: center; color: white;">
                <div class="spinner" style="border-color: rgba(255,255,255,0.3); border-top-color: white;"></div>
                <p style="margin-top: 1rem; font-size: 1.2rem;">Loading...</p>
            </div>
        `;
        document.body.appendChild(overlay);
        return overlay;
    }

    show() {
        this.overlay.style.display = 'flex';
    }

    hide() {
        this.overlay.style.display = 'none';
    }
}

// ===== Confirm Dialog =====
function confirmAction(message, callback) {
    const result = confirm(message);
    if (result && callback) {
        callback();
    }
    return result;
}

// ===== Chart Helper =====
class ChartHelper {
    static createPieChart(canvasId, data, labels) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const colors = ['#4361ee', '#06d6a0', '#f72585', '#ffd60a', '#4cc9f0'];
        
        // Simple pie chart implementation
        let total = data.reduce((a, b) => a + b, 0);
        let currentAngle = -0.5 * Math.PI;
        
        data.forEach((value, index) => {
            const sliceAngle = (2 * Math.PI * value) / total;
            
            ctx.beginPath();
            ctx.fillStyle = colors[index % colors.length];
            ctx.moveTo(canvas.width / 2, canvas.height / 2);
            ctx.arc(canvas.width / 2, canvas.height / 2, Math.min(canvas.width, canvas.height) / 2 - 10, 
                    currentAngle, currentAngle + sliceAngle);
            ctx.closePath();
            ctx.fill();
            
            currentAngle += sliceAngle;
        });
    }
}

// ===== Copy to Clipboard =====
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        window.notificationSystem.show('Copied to clipboard!', 'success', 2000);
    }).catch(() => {
        window.notificationSystem.show('Failed to copy', 'danger', 2000);
    });
}

// ===== Auto-hide alerts =====
function initAutoHideAlerts() {
    document.querySelectorAll('.alert').forEach(alert => {
        if (!alert.hasAttribute('data-persist')) {
            setTimeout(() => {
                alert.style.animation = 'slideOutRight 0.5s ease';
                setTimeout(() => alert.remove(), 500);
            }, 5000);
        }
    });
}

// ===== Initialize on DOM Ready =====
document.addEventListener('DOMContentLoaded', function() {
    // Initialize theme manager
    window.themeManager = new ThemeManager();
    
    // Initialize notification system
    window.notificationSystem = new NotificationSystem();
    
    // Initialize scroll animator
    new ScrollAnimator();
    
    // Initialize loading overlay
    window.loadingOverlay = new LoadingOverlay();
    
    // Auto-hide alerts
    initAutoHideAlerts();
    
    // Add smooth scrolling to all anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            smoothScrollTo(targetId);
        });
    });
    
    // Add ripple effect to buttons
    document.querySelectorAll('.btn').forEach(button => {
        button.addEventListener('click', function(e) {
            const ripple = document.createElement('span');
            ripple.style.cssText = `
                position: absolute;
                width: 20px;
                height: 20px;
                border-radius: 50%;
                background: rgba(255, 255, 255, 0.6);
                transform: translate(-50%, -50%);
                pointer-events: none;
                animation: ripple 0.6s ease-out;
            `;
            ripple.style.left = e.offsetX + 'px';
            ripple.style.top = e.offsetY + 'px';
            
            this.appendChild(ripple);
            setTimeout(() => ripple.remove(), 600);
        });
    });
    
    // Add animation to cards on hover
    document.querySelectorAll('.card').forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-8px) scale(1.02)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
        });
    });
    
    // Form validation for all forms
    document.querySelectorAll('form').forEach(form => {
        if (form.id) {
            new FormValidator(form.id);
        }
    });
    
    // Enhance tables
    document.querySelectorAll('table.table').forEach(table => {
        if (table.id) {
            new DataTableEnhancer(table.id);
        }
    });
    
    console.log('🚀 Smart Attendance System Initialized!');
});

// Add CSS for animations
const style = document.createElement('style');
style.textContent = `
    @keyframes ripple {
        to {
            width: 200px;
            height: 200px;
            opacity: 0;
        }
    }
    
    @keyframes slideOutRight {
        from {
            opacity: 1;
            transform: translateX(0);
        }
        to {
            opacity: 0;
            transform: translateX(100px);
        }
    }
    
    .is-invalid {
        border-color: var(--danger-color) !important;
    }
    
    .invalid-feedback {
        color: var(--danger-color);
        font-size: 0.875rem;
        margin-top: 0.25rem;
    }
`;
document.head.appendChild(style);
