// Jarvis Dashboard JavaScript

// Global state
let currentJob = null;
let currentTarget = null;

// Tab Management
function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(`${tabName}-tab`).classList.add('active');
    event.target.classList.add('active');
    
    // Load data for specific tabs
    if (tabName === 'reports') {
        refreshReports();
    } else if (tabName === 'jobs') {
        refreshJobs();
    }
}

// Form Management
function resetForm() {
    document.getElementById('analysis-form').reset();
    document.getElementById('analysis-progress').classList.add('hidden');
}

// File Upload
function uploadFile(targetField) {
    currentTarget = targetField;
    document.getElementById('file-upload').click();
}

async function handleFileUpload() {
    const fileInput = document.getElementById('file-upload');
    const file = fileInput.files[0];
    
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (response.ok) {
            document.getElementById(currentTarget).value = result.path;
            showNotification(`File uploaded: ${result.filename}`, 'success');
        } else {
            showNotification(`Upload failed: ${result.error}`, 'error');
        }
    } catch (error) {
        showNotification(`Upload error: ${error.message}`, 'error');
    }
    
    fileInput.value = '';
}

// Analysis Form Submission
document.getElementById('analysis-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Collect form data
    const config = {
        gpu1_name: document.getElementById('gpu1-name').value,
        gpu1_kineto: document.getElementById('gpu1-kineto').value,
        gpu1_et: document.getElementById('gpu1-et').value || null,
        gpu2_name: document.getElementById('gpu2-name').value,
        gpu2_kineto: document.getElementById('gpu2-kineto').value,
        gpu2_et: document.getElementById('gpu2-et').value || null,
        output_dir: document.getElementById('output-dir').value,
        api_key: document.getElementById('api-key').value || null,
        generate_plots: document.getElementById('generate-plots').checked,
        use_critical_path: document.getElementById('use-critical-path').checked
    };
    
    // Validate required fields
    if (!config.gpu1_kineto || !config.gpu2_kineto) {
        showNotification('Please provide kineto traces for both GPUs', 'error');
        return;
    }
    
    try {
        // Start analysis
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(config)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            currentJob = result.job_id;
            showNotification('Analysis started!', 'success');
            
            // Show progress section
            document.getElementById('analysis-progress').classList.remove('hidden');
            
            // Start polling for status
            pollJobStatus(result.job_id);
        } else {
            showNotification(`Error: ${result.error}`, 'error');
        }
    } catch (error) {
        showNotification(`Error starting analysis: ${error.message}`, 'error');
    }
});

// Job Status Polling
async function pollJobStatus(jobId) {
    const pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`/api/job/${jobId}`);
            const job = await response.json();
            
            if (!response.ok) {
                clearInterval(pollInterval);
                showNotification('Job not found', 'error');
                return;
            }
            
            // Update progress
            updateProgress(job);
            
            // Stop polling if completed or failed
            if (job.status === 'completed') {
                clearInterval(pollInterval);
                showNotification('Analysis completed! View in Reports tab.', 'success');
                document.getElementById('progress-fill').style.width = '100%';
            } else if (job.status === 'failed') {
                clearInterval(pollInterval);
                showNotification(`Analysis failed: ${job.error}`, 'error');
                document.getElementById('progress-fill').style.width = '100%';
                document.getElementById('progress-fill').style.background = 'var(--danger-color)';
            }
        } catch (error) {
            clearInterval(pollInterval);
            showNotification(`Error polling status: ${error.message}`, 'error');
        }
    }, 2000); // Poll every 2 seconds
}

function updateProgress(job) {
    const progressText = document.getElementById('progress-text');
    const progressDetails = document.getElementById('progress-details');
    const progressFill = document.getElementById('progress-fill');
    
    progressText.textContent = job.progress;
    
    // Update progress bar based on status
    if (job.status === 'pending') {
        progressFill.style.width = '10%';
    } else if (job.status === 'running') {
        progressFill.style.width = '50%';
    } else if (job.status === 'completed') {
        progressFill.style.width = '100%';
    }
    
    // Show details
    if (job.output_dir) {
        progressDetails.textContent = `Output directory: ${job.output_dir}`;
    }
}

// Reports Management
async function refreshReports() {
    const reportsList = document.getElementById('reports-list');
    reportsList.innerHTML = '<p class="loading">Loading reports...</p>';
    
    try {
        const response = await fetch('/api/reports');
        const reports = await response.json();
        
        if (reports.length === 0) {
            reportsList.innerHTML = '<p class="info">No reports found. Run an analysis to generate reports.</p>';
            return;
        }
        
        reportsList.innerHTML = '';
        reports.forEach(report => {
            const reportItem = document.createElement('div');
            reportItem.className = 'report-item';
            reportItem.onclick = () => viewReport(report.path);
            
            reportItem.innerHTML = `
                <h3>📊 ${report.name}</h3>
                <p>Generated: ${report.date_str}</p>
                <p>Path: ${report.path}</p>
            `;
            
            reportsList.appendChild(reportItem);
        });
    } catch (error) {
        reportsList.innerHTML = `<p class="error">Error loading reports: ${error.message}</p>`;
    }
}

async function viewReport(reportPath) {
    const reportViewer = document.getElementById('report-viewer');
    const reportContent = document.getElementById('report-content');
    
    // Show viewer
    reportViewer.classList.remove('hidden');
    reportContent.innerHTML = '<p class="loading">Loading...</p>';
    
    // Scroll to viewer
    reportViewer.scrollIntoView({ behavior: 'smooth' });
    
    try {
        const response = await fetch(`/api/report/${encodeURIComponent(reportPath)}`);
        const report = await response.json();
        
        if (!response.ok) {
            throw new Error(report.error);
        }
        
        // Display HTML content
        reportContent.innerHTML = report.html;
        
    } catch (error) {
        reportContent.innerHTML = `<p class="error">Error loading report: ${error.message}</p>`;
    }
}

function closeReportViewer() {
    document.getElementById('report-viewer').classList.add('hidden');
}

// Jobs Management
async function refreshJobs() {
    const jobsList = document.getElementById('jobs-list');
    
    if (Object.keys(window.analysisJobs || {}).length === 0) {
        jobsList.innerHTML = '<p class="info">No active jobs</p>';
        return;
    }
    
    jobsList.innerHTML = '';
    
    for (const [jobId, _] of Object.entries(window.analysisJobs || {})) {
        try {
            const response = await fetch(`/api/job/${jobId}`);
            const job = await response.json();
            
            if (response.ok) {
                const jobItem = document.createElement('div');
                jobItem.className = `job-item ${job.status}`;
                
                jobItem.innerHTML = `
                    <h3>Job: ${job.id}</h3>
                    <p><span class="job-status ${job.status}">${job.status}</span></p>
                    <p>${job.progress}</p>
                    ${job.output_dir ? `<p>Output: ${job.output_dir}</p>` : ''}
                    ${job.error ? `<p class="error">${job.error}</p>` : ''}
                `;
                
                jobsList.appendChild(jobItem);
            }
        } catch (error) {
            console.error(`Error loading job ${jobId}:`, error);
        }
    }
}

// Notifications
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 1rem 2rem;
        background: ${type === 'success' ? 'var(--success-color)' : 
                     type === 'error' ? 'var(--danger-color)' : 
                     'var(--secondary-color)'};
        color: white;
        border-radius: 6px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        z-index: 1000;
        animation: slideIn 0.3s ease;
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 5000);
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log('Jarvis Dashboard loaded');
    
    // Load initial data
    refreshReports();
});

// Add CSS for notifications
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(400px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(400px); opacity: 0; }
    }
`;
document.head.appendChild(style);
