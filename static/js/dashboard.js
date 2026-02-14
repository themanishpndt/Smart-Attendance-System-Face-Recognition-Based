// ===================================
//  Dashboard Charts & Analytics
//  Interactive Data Visualization
// ===================================

// ===== Attendance Chart =====
function createAttendanceChart(canvasId, labels, data) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Attendance',
                data: data,
                borderColor: '#4361ee',
                backgroundColor: 'rgba(67, 97, 238, 0.1)',
                borderWidth: 3,
                fill: true,
                tension: 0.4,
                pointRadius: 5,
                pointHoverRadius: 7,
                pointBackgroundColor: '#4361ee',
                pointBorderColor: '#fff',
                pointBorderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    padding: 12,
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    borderColor: '#4361ee',
                    borderWidth: 1
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
}

// ===== Class Distribution Pie Chart =====
function createClassDistribution(canvasId, labels, data) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: [
                    '#4361ee',
                    '#06d6a0',
                    '#f72585',
                    '#ffd60a',
                    '#4cc9f0',
                    '#7209b7',
                    '#ff6b35'
                ],
                borderWidth: 2,
                borderColor: '#fff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 15,
                        font: {
                            size: 12
                        }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    padding: 12
                }
            }
        }
    });
}

// ===== Weekly Attendance Bar Chart =====
function createWeeklyChart(canvasId, labels, presentData, absentData) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Present',
                    data: presentData,
                    backgroundColor: '#06d6a0',
                    borderRadius: 8
                },
                {
                    label: 'Absent',
                    data: absentData,
                    backgroundColor: '#f72585',
                    borderRadius: 8
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
}

// ===== Animated Counter =====
function animateValue(element, start, end, duration) {
    const range = end - start;
    const increment = end > start ? 1 : -1;
    const stepTime = Math.abs(Math.floor(duration / range));
    let current = start;
    
    const timer = setInterval(() => {
        current += increment;
        element.textContent = current;
        if (current === end) {
            clearInterval(timer);
        }
    }, stepTime);
}

// ===== Initialize Dashboard Stats =====
function initDashboardStats() {
    document.querySelectorAll('.stat-value').forEach(stat => {
        const value = parseInt(stat.dataset.value);
        if (!isNaN(value)) {
            animateValue(stat, 0, value, 1000);
        }
    });
}

// ===== Real-time Clock =====
function updateClock() {
    const clockElement = document.getElementById('dashboard-clock');
    if (!clockElement) return;

    const now = new Date();
    const options = { 
        hour: '2-digit', 
        minute: '2-digit', 
        second: '2-digit',
        hour12: true 
    };
    
    clockElement.textContent = now.toLocaleTimeString('en-US', options);
}

// ===== Date Display =====
function updateDate() {
    const dateElement = document.getElementById('dashboard-date');
    if (!dateElement) return;

    const now = new Date();
    const options = { 
        weekday: 'long', 
        year: 'numeric', 
        month: 'long', 
        day: 'numeric' 
    };
    
    dateElement.textContent = now.toLocaleDateString('en-US', options);
}

// ===== Progress Ring =====
class ProgressRing {
    constructor(selector, percent, color = '#4361ee') {
        this.element = document.querySelector(selector);
        if (!this.element) return;
        
        this.percent = percent;
        this.color = color;
        this.render();
    }

    render() {
        const size = 120;
        const strokeWidth = 10;
        const radius = (size - strokeWidth) / 2;
        const circumference = radius * 2 * Math.PI;
        const offset = circumference - (this.percent / 100) * circumference;

        this.element.innerHTML = `
            <svg width="${size}" height="${size}" class="progress-ring">
                <circle
                    stroke="#e0e0e0"
                    stroke-width="${strokeWidth}"
                    fill="transparent"
                    r="${radius}"
                    cx="${size / 2}"
                    cy="${size / 2}"
                />
                <circle
                    stroke="${this.color}"
                    stroke-width="${strokeWidth}"
                    fill="transparent"
                    r="${radius}"
                    cx="${size / 2}"
                    cy="${size / 2}"
                    stroke-dasharray="${circumference} ${circumference}"
                    stroke-dashoffset="${offset}"
                    stroke-linecap="round"
                    transform="rotate(-90 ${size / 2} ${size / 2})"
                    style="transition: stroke-dashoffset 0.5s ease;"
                />
                <text
                    x="50%"
                    y="50%"
                    text-anchor="middle"
                    dy="7"
                    font-size="24"
                    font-weight="bold"
                    fill="${this.color}"
                >
                    ${this.percent}%
                </text>
            </svg>
        `;
    }
}

// ===== Export Functions =====
function exportChartAsPNG(chartId, filename = 'chart.png') {
    const canvas = document.getElementById(chartId);
    if (!canvas) return;

    const url = canvas.toDataURL('image/png');
    const link = document.createElement('a');
    link.download = filename;
    link.href = url;
    link.click();
}

// ===== Initialize on Page Load =====
document.addEventListener('DOMContentLoaded', function() {
    // Initialize clock and date
    updateClock();
    updateDate();
    setInterval(updateClock, 1000);

    // Initialize dashboard stats
    initDashboardStats();

    // Initialize progress rings
    document.querySelectorAll('[data-progress-ring]').forEach(ring => {
        const percent = parseInt(ring.dataset.progressRing);
        const color = ring.dataset.color || '#4361ee';
        new ProgressRing(ring, percent, color);
    });
});

// Add CSS for progress ring
const style = document.createElement('style');
style.textContent = `
    .progress-ring {
        display: block;
        margin: 0 auto;
    }
    
    .stat-card {
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: scale(1.05);
    }
`;
document.head.appendChild(style);
