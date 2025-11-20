// IRIS Interactive Mindmap Visualization
// Built with D3.js v7 - Force-directed graph with click-to-expand

class IRISMindmap {
    constructor(containerId, data) {
        this.container = d3.select(`#${containerId}`);
        this.data = data;
        this.width = this.container.node().getBoundingClientRect().width;
        this.height = 800;
        
        // State management
        this.expandedNodes = new Set(['root']); // Root expanded by default
        this.activeNode = null;
        
        // D3 elements
        this.svg = null;
        this.g = null;
        this.simulation = null;
        this.link = null;
        this.node = null;
        
        // Initialize
        this.init();
    }
    
    init() {
        // Create SVG
        this.svg = this.container
            .append('svg')
            .attr('width', this.width)
            .attr('height', this.height)
            .attr('viewBox', [0, 0, this.width, this.height]);
        
        // Add gradient definitions
        this.addGradients();
        
        // Create main group with zoom
        this.g = this.svg.append('g');
        
        // Add zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.3, 3])
            .on('zoom', (event) => {
                this.g.attr('transform', event.transform);
            });
        
        this.svg.call(zoom);
        
        // Process data and render
        this.processData();
        this.render();
        
        // Handle window resize
        window.addEventListener('resize', () => this.handleResize());
    }
    
    addGradients() {
        const defs = this.svg.append('defs');
        
        // Gradient for root node
        const gradientRoot = defs.append('radialGradient')
            .attr('id', 'gradient-root');
        gradientRoot.append('stop')
            .attr('offset', '0%')
            .attr('stop-color', '#0066ff');
        gradientRoot.append('stop')
            .attr('offset', '100%')
            .attr('stop-color', '#0044cc');
        
        // Gradient for level 1
        const gradientLevel1 = defs.append('radialGradient')
            .attr('id', 'gradient-level1');
        gradientLevel1.append('stop')
            .attr('offset', '0%')
            .attr('stop-color', '#00d4ff');
        gradientLevel1.append('stop')
            .attr('offset', '100%')
            .attr('stop-color', '#0099cc');
        
        // Gradient for level 2
        const gradientLevel2 = defs.append('radialGradient')
            .attr('id', 'gradient-level2');
        gradientLevel2.append('stop')
            .attr('offset', '0%')
            .attr('stop-color', '#667eea');
        gradientLevel2.append('stop')
            .attr('offset', '100%')
            .attr('stop-color', '#4d5eb8');
        
        // Gradient for level 3
        const gradientLevel3 = defs.append('radialGradient')
            .attr('id', 'gradient-level3');
        gradientLevel3.append('stop')
            .attr('offset', '0%')
            .attr('stop-color', '#00b894');
        gradientLevel3.append('stop')
            .attr('offset', '100%')
            .attr('stop-color', '#008670');
    }
    
    processData() {
        // Flatten tree into nodes and links
        this.nodes = [];
        this.links = [];
        
        const traverse = (node, parent = null, depth = 0) => {
            const nodeData = {
                id: node.id,
                name: node.name,
                type: node.type || 'default',
                level: depth,
                description: node.description || '',
                details: node.details || '',
                references: node.references || '',
                metrics: node.metrics || {},
                children: node.children || [],
                _children: node.children || [], // Store original children
                parent: parent
            };
            
            this.nodes.push(nodeData);
            
            if (parent) {
                this.links.push({
                    source: parent.id,
                    target: node.id
                });
            }
            
            if (node.children) {
                node.children.forEach(child => traverse(child, nodeData, depth + 1));
            }
        };
        
        traverse(this.data);
    }
    
    getVisibleNodes() {
        const visible = new Set(['root']);
        
        const addVisible = (nodeId) => {
            const node = this.nodes.find(n => n.id === nodeId);
            if (!node) return;
            
            visible.add(nodeId);
            
            if (this.expandedNodes.has(nodeId) && node.children) {
                node.children.forEach(child => {
                    addVisible(child.id);
                });
            }
        };
        
        addVisible('root');
        
        return this.nodes.filter(n => visible.has(n.id));
    }
    
    getVisibleLinks() {
        const visibleNodeIds = new Set(this.getVisibleNodes().map(n => n.id));
        return this.links.filter(l => 
            visibleNodeIds.has(l.source.id || l.source) && 
            visibleNodeIds.has(l.target.id || l.target)
        );
    }
    
    render() {
        const visibleNodes = this.getVisibleNodes();
        const visibleLinks = this.getVisibleLinks();
        
        // Create force simulation
        this.simulation = d3.forceSimulation(visibleNodes)
            .force('link', d3.forceLink(visibleLinks)
                .id(d => d.id)
                .distance(d => {
                    const source = typeof d.source === 'object' ? d.source : visibleNodes.find(n => n.id === d.source);
                    return source.level === 0 ? 200 : 150;
                })
            )
            .force('charge', d3.forceManyBody().strength(-800))
            .force('center', d3.forceCenter(this.width / 2, this.height / 2))
            .force('collision', d3.forceCollide().radius(d => this.getNodeRadius(d) + 20));
        
        // Draw links
        this.link = this.g.selectAll('.link')
            .data(visibleLinks, d => `${d.source.id || d.source}-${d.target.id || d.target}`)
            .join(
                enter => enter.append('line')
                    .attr('class', 'link')
                    .attr('stroke', '#64748b')
                    .attr('stroke-opacity', 0.3)
                    .attr('stroke-width', 2)
                    .style('stroke-dasharray', function() {
                        return this.getTotalLength();
                    })
                    .style('stroke-dashoffset', function() {
                        return this.getTotalLength();
                    })
                    .transition()
                    .duration(800)
                    .style('stroke-dashoffset', 0),
                update => update,
                exit => exit.transition()
                    .duration(300)
                    .attr('stroke-opacity', 0)
                    .remove()
            );
        
        // Draw nodes
        const nodeGroup = this.g.selectAll('.node-group')
            .data(visibleNodes, d => d.id)
            .join(
                enter => {
                    const g = enter.append('g')
                        .attr('class', 'node-group')
                        .style('opacity', 0)
                        .call(this.drag(this.simulation));
                    
                    g.append('circle')
                        .attr('class', d => `node node-level-${d.level}`)
                        .attr('r', d => this.getNodeRadius(d))
                        .attr('fill', d => this.getNodeFill(d))
                        .attr('stroke', '#fff')
                        .attr('stroke-width', 3)
                        .style('filter', 'drop-shadow(0 4px 12px rgba(0, 0, 0, 0.2))');
                    
                    // Add expand/collapse indicator
                    g.filter(d => d.children && d.children.length > 0)
                        .append('circle')
                        .attr('class', 'expand-indicator')
                        .attr('r', 8)
                        .attr('cx', d => this.getNodeRadius(d) * 0.7)
                        .attr('cy', d => -this.getNodeRadius(d) * 0.7)
                        .attr('fill', '#fff')
                        .attr('stroke', '#0066ff')
                        .attr('stroke-width', 2);
                    
                    g.filter(d => d.children && d.children.length > 0)
                        .append('text')
                        .attr('class', 'expand-icon')
                        .attr('x', d => this.getNodeRadius(d) * 0.7)
                        .attr('y', d => -this.getNodeRadius(d) * 0.7)
                        .attr('text-anchor', 'middle')
                        .attr('dominant-baseline', 'central')
                        .attr('fill', '#0066ff')
                        .attr('font-size', '10px')
                        .attr('font-weight', 'bold')
                        .text(d => this.expandedNodes.has(d.id) ? '−' : '+');
                    
                    // Add text label
                    g.append('text')
                        .attr('class', 'node-label')
                        .attr('text-anchor', 'middle')
                        .attr('fill', '#1e293b')
                        .attr('font-size', d => d.level === 0 ? '14px' : '12px')
                        .attr('font-weight', d => d.level === 0 ? 'bold' : '600')
                        .selectAll('tspan')
                        .data(d => d.name.split('\n'))
                        .join('tspan')
                        .attr('x', 0)
                        .attr('dy', (d, i) => i === 0 ? '-0.3em' : '1.1em')
                        .text(d => d);
                    
                    return g.transition()
                        .duration(500)
                        .style('opacity', 1);
                },
                update => update,
                exit => exit.transition()
                    .duration(300)
                    .style('opacity', 0)
                    .remove()
            );
        
        this.node = nodeGroup;
        
        // Add interaction handlers
        this.node
            .on('click', (event, d) => this.handleNodeClick(event, d))
            .on('mouseenter', (event, d) => this.handleNodeHover(event, d))
            .on('mouseleave', (event, d) => this.handleNodeLeave(event, d));
        
        // Update positions on simulation tick
        this.simulation.on('tick', () => {
            this.link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
            
            this.node
                .attr('transform', d => `translate(${d.x},${d.y})`);
        });
    }
    
    getNodeRadius(node) {
        const radii = [60, 45, 35, 25];
        return radii[Math.min(node.level, 3)];
    }
    
    getNodeFill(node) {
        if (node.level === 0) return 'url(#gradient-root)';
        if (node.level === 1) return 'url(#gradient-level1)';
        if (node.level === 2) return 'url(#gradient-level2)';
        return 'url(#gradient-level3)';
    }
    
    handleNodeClick(event, d) {
        event.stopPropagation();
        
        // Toggle expand/collapse
        if (d.children && d.children.length > 0) {
            if (this.expandedNodes.has(d.id)) {
                this.expandedNodes.delete(d.id);
            } else {
                this.expandedNodes.add(d.id);
            }
            
            this.render();
        }
        
        // Show details
        this.showDetails(d);
        this.activeNode = d;
        
        // Highlight active node
        this.node.select('circle.node')
            .attr('stroke-width', n => n.id === d.id ? 5 : 3)
            .attr('stroke', n => n.id === d.id ? '#fbbf24' : '#fff');
    }
    
    handleNodeHover(event, d) {
        // Show tooltip
        const tooltip = d3.select('body')
            .append('div')
            .attr('class', 'mindmap-tooltip')
            .style('position', 'absolute')
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY + 10) + 'px')
            .style('opacity', 0);
        
        tooltip.html(`
            <strong>${d.name.replace('\n', ' ')}</strong><br/>
            <span style="font-size: 12px; color: #64748b;">${d.description}</span>
        `)
        .transition()
        .duration(200)
        .style('opacity', 1);
        
        // Store tooltip for removal
        d._tooltip = tooltip;
        
        // Highlight connections
        this.link
            .attr('stroke-opacity', l => 
                (l.source.id === d.id || l.target.id === d.id) ? 0.8 : 0.1
            )
            .attr('stroke-width', l => 
                (l.source.id === d.id || l.target.id === d.id) ? 3 : 2
            );
    }
    
    handleNodeLeave(event, d) {
        // Remove tooltip
        if (d._tooltip) {
            d._tooltip.transition()
                .duration(200)
                .style('opacity', 0)
                .remove();
            d._tooltip = null;
        }
        
        // Reset link highlighting
        this.link
            .attr('stroke-opacity', 0.3)
            .attr('stroke-width', 2);
    }
    
    showDetails(node) {
        const panel = d3.select('#node-details-panel');
        const content = d3.select('#node-details-content');
        
        let html = `
            <div class="detail-header">
                <h3>${node.name.replace('\n', ' ')}</h3>
                <span class="detail-level">Level ${node.level}</span>
            </div>
            <div class="detail-section">
                <h4>Description</h4>
                <p>${node.description}</p>
            </div>
        `;
        
        if (node.details) {
            html += `
                <div class="detail-section">
                    <h4>Details</h4>
                    <p>${node.details}</p>
                </div>
            `;
        }
        
        if (node.references) {
            html += `
                <div class="detail-section">
                    <h4>References</h4>
                    <p class="references">${node.references}</p>
                </div>
            `;
        }
        
        if (node.metrics && Object.keys(node.metrics).length > 0) {
            html += `<div class="detail-section"><h4>Key Metrics</h4><div class="metrics-grid">`;
            for (const [key, value] of Object.entries(node.metrics)) {
                html += `
                    <div class="metric-item">
                        <span class="metric-label">${key}</span>
                        <span class="metric-value">${value}</span>
                    </div>
                `;
            }
            html += `</div></div>`;
        }
        
        if (node.children && node.children.length > 0) {
            html += `
                <div class="detail-section">
                    <h4>Child Nodes (${node.children.length})</h4>
                    <div class="children-list">
                        ${node.children.map(c => `<span class="child-tag">${c.name}</span>`).join('')}
                    </div>
                </div>
            `;
        }
        
        content.html(html);
        panel.classed('active', true);
    }
    
    hideDetails() {
        d3.select('#node-details-panel').classed('active', false);
        
        // Reset active node highlight
        this.node.select('circle.node')
            .attr('stroke-width', 3)
            .attr('stroke', '#fff');
        
        this.activeNode = null;
    }
    
    expandAll() {
        this.nodes.forEach(n => {
            if (n.children && n.children.length > 0) {
                this.expandedNodes.add(n.id);
            }
        });
        this.render();
    }
    
    collapseAll() {
        this.expandedNodes.clear();
        this.expandedNodes.add('root'); // Keep root expanded
        this.render();
    }
    
    resetView() {
        this.svg.transition()
            .duration(750)
            .call(
                d3.zoom().transform,
                d3.zoomIdentity.translate(0, 0).scale(1)
            );
    }
    
    drag(simulation) {
        function dragstarted(event) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        }
        
        function dragged(event) {
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }
        
        function dragended(event) {
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }
        
        return d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended);
    }
    
    handleResize() {
        this.width = this.container.node().getBoundingClientRect().width;
        this.svg.attr('width', this.width);
        this.svg.attr('viewBox', [0, 0, this.width, this.height]);
        
        if (this.simulation) {
            this.simulation.force('center', d3.forceCenter(this.width / 2, this.height / 2));
            this.simulation.alpha(0.3).restart();
        }
    }
}

// Initialize mindmap when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Create mindmap instance
    const mindmap = new IRISMindmap('mindmap-container', mindmapData);
    
    // Wire up control buttons
    document.getElementById('expand-all-btn').addEventListener('click', () => {
        mindmap.expandAll();
    });
    
    document.getElementById('collapse-all-btn').addEventListener('click', () => {
        mindmap.collapseAll();
    });
    
    document.getElementById('reset-view-btn').addEventListener('click', () => {
        mindmap.resetView();
    });
    
    document.getElementById('close-panel-btn').addEventListener('click', () => {
        mindmap.hideDetails();
    });
    
    // Store global reference for console access
    window.irisMindmap = mindmap;
});
