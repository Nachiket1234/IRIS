// Demo Page JavaScript
// Handles file uploads, strategy selection, and result visualization

(function() {
    'use strict';
    
    let supportFiles = [];
    let queryFile = null;
    let selectedStrategy = 'memory';
    
    // DOM Elements
    const supportUpload = document.getElementById('support-upload');
    const supportFileInput = document.getElementById('support-file-input');
    const supportFilesDisplay = document.getElementById('support-files');
    
    const queryUpload = document.getElementById('query-upload');
    const queryFileInput = document.getElementById('query-file-input');
    const queryFileDisplay = document.getElementById('query-file');
    
    const runButton = document.getElementById('run-segmentation');
    const placeholder = document.getElementById('placeholder');
    const results = document.getElementById('results');
    const loading = document.getElementById('loading');
    
    // Support Set Upload
    supportUpload.addEventListener('click', () => {
        supportFileInput.click();
    });
    
    supportUpload.addEventListener('dragover', (e) => {
        e.preventDefault();
        supportUpload.classList.add('active');
    });
    
    supportUpload.addEventListener('dragleave', () => {
        supportUpload.classList.remove('active');
    });
    
    supportUpload.addEventListener('drop', (e) => {
        e.preventDefault();
        supportUpload.classList.remove('active');
        handleSupportFiles(e.dataTransfer.files);
    });
    
    supportFileInput.addEventListener('change', (e) => {
        handleSupportFiles(e.target.files);
    });
    
    function handleSupportFiles(files) {
        // Limit to 3 files
        const newFiles = Array.from(files).slice(0, 3 - supportFiles.length);
        supportFiles = [...supportFiles, ...newFiles];
        
        updateSupportFilesDisplay();
    }
    
    function updateSupportFilesDisplay() {
        supportFilesDisplay.innerHTML = '';
        
        supportFiles.forEach((file, index) => {
            const fileItem = document.createElement('div');
            fileItem.className = 'file-item';
            fileItem.innerHTML = `
                <span class="file-name">📄 ${file.name}</span>
                <button class="file-remove" data-index="${index}">×</button>
            `;
            supportFilesDisplay.appendChild(fileItem);
        });
        
        // Add remove handlers
        document.querySelectorAll('.file-remove').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const index = parseInt(e.target.dataset.index);
                supportFiles.splice(index, 1);
                updateSupportFilesDisplay();
            });
        });
    }
    
    // Query Upload
    queryUpload.addEventListener('click', () => {
        queryFileInput.click();
    });
    
    queryUpload.addEventListener('dragover', (e) => {
        e.preventDefault();
        queryUpload.classList.add('active');
    });
    
    queryUpload.addEventListener('dragleave', () => {
        queryUpload.classList.remove('active');
    });
    
    queryUpload.addEventListener('drop', (e) => {
        e.preventDefault();
        queryUpload.classList.remove('active');
        handleQueryFile(e.dataTransfer.files[0]);
    });
    
    queryFileInput.addEventListener('change', (e) => {
        handleQueryFile(e.target.files[0]);
    });
    
    function handleQueryFile(file) {
        if (!file) return;
        
        queryFile = file;
        queryFileDisplay.innerHTML = `
            <div class="file-item">
                <span class="file-name">🖼️ ${file.name}</span>
                <button class="file-remove" id="remove-query">×</button>
            </div>
        `;
        
        document.getElementById('remove-query').addEventListener('click', () => {
            queryFile = null;
            queryFileDisplay.innerHTML = '';
        });
    }
    
    // Strategy Selection
    document.querySelectorAll('input[name="strategy"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            selectedStrategy = e.target.value;
        });
    });
    
    // Example Datasets
    document.querySelectorAll('.btn-example').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const dataset = e.target.dataset.dataset;
            loadExampleDataset(dataset);
        });
    });
    
    function loadExampleDataset(dataset) {
        // In a real implementation, this would load example images
        console.log(`Loading example dataset: ${dataset}`);
        
        // Mock example data
        const examples = {
            'chest-xray': {
                support: ['chest_support_1.jpg', 'chest_support_2.jpg'],
                query: 'chest_query.jpg',
                strategy: 'ensemble'
            },
            'isic': {
                support: ['isic_support_1.jpg', 'isic_support_2.jpg', 'isic_support_3.jpg'],
                query: 'isic_query.jpg',
                strategy: 'ensemble'
            },
            'brain-mri': {
                support: ['brain_support_1.jpg'],
                query: 'brain_query.jpg',
                strategy: 'one-shot'
            }
        };
        
        const example = examples[dataset];
        if (example) {
            // Set strategy
            document.querySelector(`input[value="${example.strategy}"]`).checked = true;
            selectedStrategy = example.strategy;
            
            // Show notification
            showNotification(`Loaded ${dataset} example dataset`, 'success');
        }
    }
    
    // Run Segmentation
    runButton.addEventListener('click', () => {
        if (!queryFile) {
            showNotification('Please upload a query image', 'error');
            return;
        }
        
        if (selectedStrategy !== 'memory' && supportFiles.length === 0) {
            showNotification('Please upload support images', 'error');
            return;
        }
        
        runSegmentation();
    });
    
    function runSegmentation() {
        // Hide placeholder, show loading
        placeholder.style.display = 'none';
        results.style.display = 'none';
        loading.style.display = 'block';
        
        // Simulate processing steps
        const steps = [
            'Initializing model...',
            'Processing support set...',
            'Encoding task embeddings...',
            'Running inference...',
            'Generating visualizations...'
        ];
        
        let stepIndex = 0;
        const loadingDetail = document.getElementById('loading-detail');
        
        const stepInterval = setInterval(() => {
            if (stepIndex < steps.length) {
                loadingDetail.textContent = steps[stepIndex];
                stepIndex++;
            } else {
                clearInterval(stepInterval);
                showResults();
            }
        }, 600);
    }
    
    function showResults() {
        loading.style.display = 'none';
        results.style.display = 'block';
        
        // Generate mock results
        const mockResults = generateMockResults();
        displayResults(mockResults);
    }
    
    function generateMockResults() {
        // Mock performance based on strategy
        const strategyMetrics = {
            'memory': { dice: 0.843, iou: 0.729, time: 35, confidence: 0.91 },
            'one-shot': { dice: 0.815, iou: 0.687, time: 52, confidence: 0.87 },
            'ensemble': { dice: 0.851, iou: 0.741, time: 78, confidence: 0.93 },
            'tuning': { dice: 0.873, iou: 0.774, time: 2100, confidence: 0.95 }
        };
        
        return strategyMetrics[selectedStrategy];
    }
    
    function displayResults(metrics) {
        // Update metrics display
        document.getElementById('dice-score').textContent = (metrics.dice * 100).toFixed(1) + '%';
        document.getElementById('iou-score').textContent = (metrics.iou * 100).toFixed(1) + '%';
        document.getElementById('inference-time').textContent = metrics.time < 1000 
            ? metrics.time + 'ms' 
            : (metrics.time / 1000).toFixed(1) + 's';
        document.getElementById('confidence').textContent = (metrics.confidence * 100).toFixed(1) + '%';
        
        // In a real implementation, this would render actual segmentation results
        // For demo purposes, we'll create placeholder canvases
        renderMockVisualizations();
    }
    
    function renderMockVisualizations() {
        // This would normally render actual segmentation results
        // For demo, we'll show placeholder graphics
        
        const canvases = [
            'canvas-overlay',
            'canvas-original',
            'canvas-segmented',
            'canvas-mask',
            'canvas-confidence'
        ];
        
        canvases.forEach(id => {
            const canvas = document.getElementById(id);
            if (canvas) {
                const ctx = canvas.getContext('2d');
                canvas.width = 512;
                canvas.height = 512;
                
                // Draw placeholder
                ctx.fillStyle = '#f1f5f9';
                ctx.fillRect(0, 0, 512, 512);
                
                ctx.fillStyle = '#64748b';
                ctx.font = '18px Inter';
                ctx.textAlign = 'center';
                ctx.fillText('Demo Visualization', 256, 256);
                ctx.font = '14px Inter';
                ctx.fillText('(Connect to backend for real results)', 256, 286);
            }
        });
    }
    
    // Visualization Tabs
    document.querySelectorAll('.viz-tab').forEach(tab => {
        tab.addEventListener('click', (e) => {
            const targetTab = e.target.dataset.tab;
            
            // Update active tab
            document.querySelectorAll('.viz-tab').forEach(t => t.classList.remove('active'));
            e.target.classList.add('active');
            
            // Update active panel
            document.querySelectorAll('.viz-panel').forEach(p => p.classList.remove('active'));
            document.getElementById(`tab-${targetTab}`).classList.add('active');
        });
    });
    
    // Download Results
    document.getElementById('download-results').addEventListener('click', () => {
        // In real implementation, would package and download results
        showNotification('Results downloaded successfully', 'success');
    });
    
    // Notification System
    function showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 80px;
            right: 20px;
            padding: 1rem 1.5rem;
            background: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#0066ff'};
            color: white;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            z-index: 10000;
            animation: slideIn 0.3s ease;
            font-family: Inter;
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => {
                notification.remove();
            }, 300);
        }, 3000);
    }
    
    // Add CSS animations
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideIn {
            from {
                transform: translateX(400px);
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
                transform: translateX(400px);
                opacity: 0;
            }
        }
    `;
    document.head.appendChild(style);
    
})();
