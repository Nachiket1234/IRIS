# IRIS Medical Segmentation - Website

Professional website for IRIS (Interactive and Refined Image Segmentation) medical AI framework.

## 🌐 Website Structure

```
docs/website/
├── index.html              # Landing page with interactive mindmap
├── metrics.html            # Performance benchmarks and analytics
├── demo.html               # Live demo interface
└── assets/
    ├── css/
    │   ├── main.css        # Core design system and shared styles
    │   ├── mindmap.css     # Mindmap visualization styles
    │   ├── metrics.css     # Metrics page styles
    │   └── demo.css        # Demo page styles
    ├── js/
    │   ├── main.js         # General page interactions
    │   ├── mindmap-data.js # IRIS architecture data structure
    │   ├── mindmap.js      # D3.js interactive mindmap
    │   ├── metrics-charts.js # Chart.js performance visualizations
    │   └── demo.js         # Demo page functionality
    └── images/             # Image assets (empty - add as needed)
```

## ✨ Features

### Landing Page (`index.html`)
- **Hero Section**: Animated gradient background with key statistics
  - 85.1% Dice Score
  - 9 Datasets
  - 6 Modalities
  - 46% Energy Efficiency
- **Features Grid**: Showcases core capabilities
  - Few-Shot Learning
  - Multi-Modal Support
  - Rapid Adaptation
  - Memory Bank
- **Interactive Mindmap**: D3.js force-directed graph
  - Click-to-expand/collapse nodes
  - 5 main architecture branches
  - Details panel with metrics
  - Zoom/pan controls
  - Drag nodes
- **Call-to-Action**: Links to demo and documentation

### Metrics Page (`metrics.html`)
- **Overall Performance**: Key metrics cards
- **Per-Dataset Performance**: 
  - Interactive bar chart (Chart.js)
  - Detailed comparison table
  - 9 datasets across 6 modalities
- **Strategy Comparison**: 
  - 4 inference strategies
  - Performance/speed trade-offs
  - Use case recommendations
- **Ablation Studies**: Component contribution analysis
- **Cross-Modality**: Radar chart showing generalization

### Demo Page (`demo.html`)
- **Upload Interface**: 
  - Drag-and-drop support set (1-3 images)
  - Drag-and-drop query image
  - File management
- **Strategy Selection**: 
  - Radio buttons for 4 strategies
  - Speed/accuracy indicators
- **Results Visualization**: 
  - 4 visualization modes (Overlay, Side-by-Side, Mask, Confidence)
  - Performance metrics (Dice, IoU, Time, Confidence)
  - Download results
- **Example Datasets**: Pre-loaded examples
  - Chest X-Ray
  - Skin Lesion (ISIC)
  - Brain MRI

## 🎨 Design System

### Colors
- **Primary**: `#0066ff` (Blue)
- **Secondary**: `#00d4ff` (Cyan)
- **Accent**: `#667eea` (Purple)
- **Success**: `#00b894` (Teal)
- **Text**: `#1e293b` (Slate)
- **Muted**: `#64748b` (Gray)

### Typography
- **Font**: Inter (sans-serif) + Fira Code (monospace)
- **Weights**: 300, 400, 500, 600, 700, 800

### Components
- Glassmorphism navigation
- Gradient buttons
- Card-based layouts
- Smooth animations
- Responsive design (mobile-first)

## 🚀 Getting Started

### Local Development

1. **No build required** - Pure HTML/CSS/JavaScript
   
2. **Open in browser**:
   ```bash
   # Navigate to website directory
   cd docs/website
   
   # Open with Python server (recommended)
   python -m http.server 8000
   
   # Or use VS Code Live Server extension
   # Right-click index.html > Open with Live Server
   ```

3. **Access**:
   - Landing: `http://localhost:8000/index.html`
   - Metrics: `http://localhost:8000/metrics.html`
   - Demo: `http://localhost:8000/demo.html`

### Dependencies (CDN)
- **D3.js v7**: Interactive mindmap visualization
- **Chart.js v4**: Performance charts and graphs
- **Google Fonts**: Inter and Fira Code

## 📊 Mindmap Data Structure

The mindmap (`mindmap-data.js`) contains hierarchical IRIS architecture:

```javascript
{
    id: "root",
    name: "IRIS Medical Segmentation",
    children: [
        { id: "purpose", ... },        // Purpose & Goals
        { id: "architecture", ... },   // 5 Core Components
        { id: "inference", ... },      // 4 Strategies
        { id: "validation", ... },     // 9 Datasets
        { id: "implementation", ... }  // Tech Stack
    ]
}
```

### Interaction Features
- **Click node**: Toggle expand/collapse children
- **Hover node**: Show tooltip with description
- **Click node**: Show detailed panel with metrics/references
- **Drag node**: Reposition in force simulation
- **Zoom/Pan**: Mouse wheel and drag background
- **Controls**: Expand All, Collapse All, Reset View

## 📈 Charts & Visualizations

### Metrics Page Charts
1. **Dataset Performance** (Bar Chart)
   - 3 strategies × 9 datasets
   - Shows Dice score progression

2. **Ablation Study** (Horizontal Bar)
   - Cumulative component effects
   - Baseline → Full Model

3. **Cross-Modality** (Radar Chart)
   - 6 modalities comparison
   - Strategy performance overlay

## 🎯 Demo Functionality

### Current Implementation
- **Frontend Only**: Mock results and visualizations
- **File Uploads**: Accepts images but doesn't process
- **Strategy Selection**: Updates UI accordingly
- **Mock Results**: Generates realistic performance metrics

### Backend Integration (Future)
To connect to real IRIS backend:

1. Update `demo.js` `runSegmentation()` function
2. Add API endpoint calls
3. Handle image preprocessing
4. Stream results back to frontend
5. Render actual segmentation masks

Example API structure:
```javascript
const response = await fetch('/api/segment', {
    method: 'POST',
    body: formData,
    headers: { 'Content-Type': 'multipart/form-data' }
});
const results = await response.json();
// results = { mask, dice, iou, time, confidence }
```

## 🔧 Customization

### Adding New Dataset
1. Update `mindmap-data.js` validation section
2. Add to performance table in `metrics.html`
3. Update chart data in `metrics-charts.js`

### Changing Colors
Edit CSS custom properties in `main.css`:
```css
:root {
    --primary-color: #0066ff;
    --secondary-color: #00d4ff;
    /* ... */
}
```

### Adding Pages
1. Create new HTML file in `docs/website/`
2. Link stylesheet: `<link rel="stylesheet" href="assets/css/main.css">`
3. Add navigation link in all pages
4. Include footer from existing pages

## 📱 Responsive Breakpoints

- **Mobile**: < 768px
- **Tablet**: 768px - 1024px
- **Desktop**: > 1024px

All pages are fully responsive with:
- Flexible grids
- Mobile navigation
- Touch-friendly controls
- Optimized font sizes

## 🌟 Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Mobile browsers (iOS Safari, Chrome Mobile)

**Required Features**:
- CSS Grid
- CSS Custom Properties
- ES6 JavaScript
- SVG support (for D3.js)
- Canvas API (for Chart.js)

## 📝 Content Updates

### Update Statistics
Edit `index.html` hero section:
```html
<div class="stat-card">
    <div class="stat-value" data-value="85.1">85.1</div>
    <div class="stat-label">Dice Score</div>
</div>
```

### Update Documentation Links
Footer section in all HTML files:
```html
<div class="footer-section">
    <h4>Documentation</h4>
    <ul>
        <li><a href="#">Architecture Guide</a></li>
        <!-- Add more links -->
    </ul>
</div>
```

## 🚢 Deployment

### GitHub Pages
```bash
# Repository settings > Pages
# Source: docs/website/ directory
# Custom domain (optional): iris.example.com
```

### Netlify/Vercel
```bash
# Build command: (none needed)
# Publish directory: docs/website
# No build process required
```

### Traditional Hosting
Upload `docs/website/` contents to web server root or subdirectory.

## 📄 License

Part of IRIS Medical Segmentation project. See main repository for license details.

## 🙏 Credits

- **D3.js**: Mike Bostock and contributors
- **Chart.js**: Chart.js team
- **Google Fonts**: Inter by Rasmus Andersson, Fira Code
- **Icons**: Unicode emoji characters

---

**Built with ❤️ for advancing medical AI**
