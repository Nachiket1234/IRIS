// Metrics Page Chart Configurations
// Chart.js visualizations for IRIS performance data

document.addEventListener('DOMContentLoaded', () => {
    
    // Dataset Performance Chart
    const datasetCtx = document.getElementById('datasetChart');
    if (datasetCtx) {
        new Chart(datasetCtx, {
            type: 'bar',
            data: {
                labels: [
                    'Chest X-Ray',
                    'ISIC',
                    'Brain MRI',
                    'DRIVE',
                    'Kvasir',
                    'AMOS',
                    'SegTHOR',
                    'COVID CT',
                    'Pancreas'
                ],
                datasets: [
                    {
                        label: 'One-Shot (K=1)',
                        data: [91.2, 86.4, 78.3, 79.5, 83.1, 80.7, 77.9, 82.4, 75.8],
                        backgroundColor: 'rgba(100, 116, 139, 0.7)',
                        borderColor: 'rgba(100, 116, 139, 1)',
                        borderWidth: 2
                    },
                    {
                        label: 'Ensemble (K=3)',
                        data: [93.8, 89.7, 82.1, 82.8, 86.4, 84.2, 81.3, 85.6, 79.4],
                        backgroundColor: 'rgba(0, 102, 255, 0.7)',
                        borderColor: 'rgba(0, 102, 255, 1)',
                        borderWidth: 2
                    },
                    {
                        label: 'In-Context Tuning',
                        data: [95.1, 91.2, 84.6, 85.3, 88.9, 86.8, 83.7, 87.9, 82.1],
                        backgroundColor: 'rgba(102, 126, 234, 0.7)',
                        borderColor: 'rgba(102, 126, 234, 1)',
                        borderWidth: 2
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            font: {
                                size: 13,
                                family: 'Inter'
                            },
                            padding: 15
                        }
                    },
                    title: {
                        display: true,
                        text: 'Dice Score (%) by Dataset and Strategy',
                        font: {
                            size: 16,
                            weight: '600',
                            family: 'Inter'
                        },
                        padding: 20
                    },
                    tooltip: {
                        backgroundColor: 'rgba(30, 41, 59, 0.95)',
                        titleFont: {
                            size: 14,
                            family: 'Inter'
                        },
                        bodyFont: {
                            size: 13,
                            family: 'Inter'
                        },
                        padding: 12,
                        cornerRadius: 8,
                        callbacks: {
                            label: function(context) {
                                return context.dataset.label + ': ' + context.parsed.y.toFixed(1) + '%';
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            },
                            font: {
                                size: 12,
                                family: 'Inter'
                            }
                        },
                        grid: {
                            color: 'rgba(0, 0, 0, 0.05)'
                        }
                    },
                    x: {
                        ticks: {
                            font: {
                                size: 11,
                                family: 'Inter'
                            }
                        },
                        grid: {
                            display: false
                        }
                    }
                }
            }
        });
    }
    
    // Ablation Study Chart
    const ablationCtx = document.getElementById('ablationChart');
    if (ablationCtx) {
        new Chart(ablationCtx, {
            type: 'bar',
            data: {
                labels: [
                    'Baseline (No Components)',
                    '+ Task Encoding Module',
                    '+ Bidirectional Attention',
                    '+ Memory Bank',
                    '+ FiLM Conditioning',
                    'Full IRIS Model'
                ],
                datasets: [{
                    label: 'Mean Dice Score (%)',
                    data: [72.8, 77.0, 79.8, 81.7, 84.8, 85.1],
                    backgroundColor: [
                        'rgba(100, 116, 139, 0.7)',
                        'rgba(0, 212, 255, 0.7)',
                        'rgba(0, 184, 148, 0.7)',
                        'rgba(102, 126, 234, 0.7)',
                        'rgba(0, 102, 255, 0.7)',
                        'rgba(0, 102, 255, 1)'
                    ],
                    borderColor: [
                        'rgba(100, 116, 139, 1)',
                        'rgba(0, 212, 255, 1)',
                        'rgba(0, 184, 148, 1)',
                        'rgba(102, 126, 234, 1)',
                        'rgba(0, 102, 255, 1)',
                        'rgba(0, 102, 255, 1)'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: false
                    },
                    title: {
                        display: true,
                        text: 'Component Contribution (Cumulative Effect)',
                        font: {
                            size: 16,
                            weight: '600',
                            family: 'Inter'
                        },
                        padding: 20
                    },
                    tooltip: {
                        backgroundColor: 'rgba(30, 41, 59, 0.95)',
                        titleFont: {
                            size: 14,
                            family: 'Inter'
                        },
                        bodyFont: {
                            size: 13,
                            family: 'Inter'
                        },
                        padding: 12,
                        cornerRadius: 8,
                        callbacks: {
                            label: function(context) {
                                return 'Dice Score: ' + context.parsed.x.toFixed(1) + '%';
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            },
                            font: {
                                size: 12,
                                family: 'Inter'
                            }
                        },
                        grid: {
                            color: 'rgba(0, 0, 0, 0.05)'
                        }
                    },
                    y: {
                        ticks: {
                            font: {
                                size: 12,
                                family: 'Inter'
                            }
                        },
                        grid: {
                            display: false
                        }
                    }
                }
            }
        });
    }
    
    // Modality Performance Chart
    const modalityCtx = document.getElementById('modalityChart');
    if (modalityCtx) {
        new Chart(modalityCtx, {
            type: 'radar',
            data: {
                labels: [
                    'X-Ray (Chest)',
                    'Dermoscopy (ISIC)',
                    'MRI (Brain)',
                    'CT (Multi-Organ)',
                    'Fundoscopy (DRIVE)',
                    'Endoscopy (Kvasir)'
                ],
                datasets: [
                    {
                        label: 'One-Shot (K=1)',
                        data: [91.2, 86.4, 78.3, 80.7, 79.5, 83.1],
                        borderColor: 'rgba(100, 116, 139, 1)',
                        backgroundColor: 'rgba(100, 116, 139, 0.2)',
                        borderWidth: 2,
                        pointBackgroundColor: 'rgba(100, 116, 139, 1)',
                        pointRadius: 4
                    },
                    {
                        label: 'Ensemble (K=3)',
                        data: [93.8, 89.7, 82.1, 84.2, 82.8, 86.4],
                        borderColor: 'rgba(0, 102, 255, 1)',
                        backgroundColor: 'rgba(0, 102, 255, 0.2)',
                        borderWidth: 2,
                        pointBackgroundColor: 'rgba(0, 102, 255, 1)',
                        pointRadius: 4
                    },
                    {
                        label: 'In-Context Tuning',
                        data: [95.1, 91.2, 84.6, 86.8, 85.3, 88.9],
                        borderColor: 'rgba(102, 126, 234, 1)',
                        backgroundColor: 'rgba(102, 126, 234, 0.2)',
                        borderWidth: 2,
                        pointBackgroundColor: 'rgba(102, 126, 234, 1)',
                        pointRadius: 4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            font: {
                                size: 13,
                                family: 'Inter'
                            },
                            padding: 15
                        }
                    },
                    title: {
                        display: true,
                        text: 'Cross-Modality Performance (Dice Score %)',
                        font: {
                            size: 16,
                            weight: '600',
                            family: 'Inter'
                        },
                        padding: 20
                    },
                    tooltip: {
                        backgroundColor: 'rgba(30, 41, 59, 0.95)',
                        titleFont: {
                            size: 14,
                            family: 'Inter'
                        },
                        bodyFont: {
                            size: 13,
                            family: 'Inter'
                        },
                        padding: 12,
                        cornerRadius: 8,
                        callbacks: {
                            label: function(context) {
                                return context.dataset.label + ': ' + context.parsed.r.toFixed(1) + '%';
                            }
                        }
                    }
                },
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            stepSize: 20,
                            callback: function(value) {
                                return value + '%';
                            },
                            font: {
                                size: 11,
                                family: 'Inter'
                            }
                        },
                        pointLabels: {
                            font: {
                                size: 12,
                                family: 'Inter',
                                weight: '600'
                            }
                        },
                        grid: {
                            color: 'rgba(0, 0, 0, 0.05)'
                        }
                    }
                }
            }
        });
    }
    
});
