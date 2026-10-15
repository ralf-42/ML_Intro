let regMode = 'separate';
let l1Lambda = 0;
let l2Lambda = 0;
let elasticLambda = 0;
let alpha = 0.5;

const layers = [
    {id: 'input', nodes: 3},
    {id: 'hidden', nodes: 4},
    {id: 'output', nodes: 2}
];

const baseWeights = [
    [0.8, 0.5, 0.9, 0.6],
    [0.7, 0.4, 0.6, 0.8],
    [0.6, 0.7, 0.7, 0.5]
];

let weights = [];

function setup() {
    const canvas = createCanvas(800, 600);
    canvas.parent('canvas-container');
    
    // Initialize weights
    weights = calculateWeights();
    
    // Setup event listeners
    setupEventListeners();
}

function draw() {
    background(255);
    drawNetwork();
}

function calculateWeights() {
    if (regMode === 'separate') {
        return baseWeights.map(row =>
            row.map(w => {
                let weight = w;
                if (l1Lambda > 0) {
                    weight = Math.abs(w) <= l1Lambda ? 0 : 
                            (Math.abs(w) - l1Lambda) * Math.sign(w);
                }
                if (l2Lambda > 0) {
                    weight = weight / (1 + l2Lambda);
                }
                return weight;
            })
        );
    } else {
        return baseWeights.map(row =>
            row.map(w => {
                let weight = w;
                const l1Effect = alpha * elasticLambda;
                const l2Effect = (1 - alpha) * elasticLambda;
                
                if (l1Effect > 0) {
                    weight = Math.abs(w) <= l1Effect ? 0 : 
                            (Math.abs(w) - l1Effect) * Math.sign(w);
                }
                if (l2Effect > 0) {
                    weight = weight / (1 + l2Effect);
                }
                return weight;
            })
        );
    }
}

function getNodeColor(layer) {
    switch(layer) {
        case 'input': return color('#4299e1');
        case 'hidden': return color('#48bb78');
        case 'output': return color('#ed64a6');
        default: return color('#a0aec0');
    }
}

function getWeightVisualization(originalWeight, currentWeight) {
    const isZero = Math.abs(currentWeight) < 0.01;
    if (isZero) {
        return {
            color: color('#ddd'),
            weight: 1,
            opacity: 0.3,
            dashed: true
        };
    }
    
    const weightReduction = Math.abs(currentWeight / originalWeight);
    const opacity = Math.max(0.2, weightReduction);
    
    return {
        color: currentWeight > 0 ? color('#22c55e') : color('#ef4444'),
        weight: Math.max(1, Math.abs(currentWeight) * 4),
        opacity: opacity,
        dashed: false
    };
}

function drawNetwork() {
    // Draw connections
    for (let layerIdx = 0; layerIdx < layers.length - 1; layerIdx++) {
        const layer = layers[layerIdx];
        const nextLayer = layers[layerIdx + 1];
        const startX = 150 + layerIdx * 250;
        const endX = startX + 250;
        
        for (let i = 0; i < layer.nodes; i++) {
            const startY = 150 + (i * 120);
            
            for (let j = 0; j < nextLayer.nodes; j++) {
                const endY = 150 + (j * 120);
                const originalWeight = baseWeights[i]?.[j] ?? 0;
                const currentWeight = weights[i]?.[j] ?? 0;
                const viz = getWeightVisualization(originalWeight, currentWeight);
                
                push();
                stroke(viz.color);
                strokeWeight(viz.weight);
                if (viz.dashed) {
                    drawingContext.setLineDash([5, 5]);
                } else {
                    drawingContext.setLineDash([]);
                }
                line(startX, startY, endX, endY);
                pop();
                
                if (Math.abs(currentWeight) < 0.01) {
                    push();
                    fill(102);
                    noStroke();
                    textSize(10);
                    textAlign(CENTER, CENTER);
                    text('0', (startX + endX) / 2, (startY + endY) / 2);
                    pop();
                }
            }
        }
    }
    
    // Draw nodes
    for (let layerIdx = 0; layerIdx < layers.length; layerIdx++) {
        const layer = layers[layerIdx];
        const x = 150 + layerIdx * 250;
        
        for (let i = 0; i < layer.nodes; i++) {
            const y = 150 + (i * 120);
            
            // Draw node
            push();
            fill(getNodeColor(layer.id));
            noStroke();
            ellipse(x, y, 40, 40);
            
            // Draw label
            fill(255);
            noStroke();
            textAlign(CENTER, CENTER);
            textSize(12);
            let label = layer.id === 'input' ? `x${i+1}` :
                       layer.id === 'output' ? `y${i+1}` : `h${i+1}`;
            text(label, x, y);
            pop();
        }
    }
}

function setupEventListeners() {
    // Separate controls
    document.getElementById('l1-slider').addEventListener('input', function(e) {
        l1Lambda = parseFloat(e.target.value);
        document.getElementById('l1-value').textContent = l1Lambda.toFixed(2);
        weights = calculateWeights();
    });
    
    document.getElementById('l2-slider').addEventListener('input', function(e) {
        l2Lambda = parseFloat(e.target.value);
        document.getElementById('l2-value').textContent = l2Lambda.toFixed(2);
        weights = calculateWeights();
    });
    
    // Elastic Net controls
    document.getElementById('elastic-slider').addEventListener('input', function(e) {
        elasticLambda = parseFloat(e.target.value);
        document.getElementById('elastic-value').textContent = elasticLambda.toFixed(2);
        weights = calculateWeights();
    });
    
    document.getElementById('alpha-slider').addEventListener('input', function(e) {
        alpha = parseFloat(e.target.value);
        document.getElementById('alpha-value').textContent = alpha.toFixed(2);
        weights = calculateWeights();
    });
}

function setRegMode(mode) {
    regMode = mode;
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelector(`button[onclick="setRegMode('${mode}')"]`).classList.add('active');
    
    if (mode === 'separate') {
        document.getElementById('separate-controls').style.display = 'block';
        document.getElementById('elastic-controls').style.display = 'none';
    } else {
        document.getElementById('separate-controls').style.display = 'none';
        document.getElementById('elastic-controls').style.display = 'block';
    }
    
    weights = calculateWeights();
}