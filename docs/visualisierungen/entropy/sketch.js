// Globale Variablen
let confidence = 0.5;
let points = [];
let slider;
const numPoints = 100;
const margins = {
    top: 50,
    right: 50,
    bottom: 100,
    left: 50
};

// Hilfsfunktionen zuerst definieren
function generatePoints() {
    points = [];
    
    // Klasse A (rot)
    for (let i = 0; i < numPoints; i++) {
        const noise = (random() - 0.5) * (1 - confidence) * 2;
        points.push({
            x: i / numPoints,
            y: confidence + noise,
            class: 'A'
        });
    }
    
    // Klasse B (blau)
    for (let i = 0; i < numPoints; i++) {
        const noise = (random() - 0.5) * (1 - confidence) * 2;
        points.push({
            x: i / numPoints,
            y: (1 - confidence) + noise,
            class: 'B'
        });
    }
}

function calculateEntropy(p) {
    if (p === 0 || p === 1) return 0;
    return -(p * log(p) / log(2) + (1-p) * log(1-p) / log(2));
}

function drawGrid() {
    const graphHeight = height - margins.top - margins.bottom;
    const graphWidth = width - margins.left - margins.right;
    
    stroke(200);
    strokeWeight(1);
    
    // Vertikale Linien
    for (let x = margins.left; x <= width - margins.right; x += graphWidth/10) {
        line(x, margins.top, x, height - margins.bottom);
    }
    
    // Horizontale Linien
    for (let y = margins.top; y <= height - margins.bottom; y += graphHeight/10) {
        line(margins.left, y, width - margins.right, y);
    }
    
    // Achsen
    stroke(0);
    strokeWeight(2);
    // X-Achse
    line(margins.left, height - margins.bottom, width - margins.right, height - margins.bottom);
    // Y-Achse
    line(margins.left, margins.top, margins.left, height - margins.bottom);
}

function drawPoints() {
    const graphHeight = height - margins.top - margins.bottom;
    
    for (let point of points) {
        if (point.class === 'A') {
            fill(239, 68, 68, 150); // Rot
        } else {
            fill(59, 130, 246, 150); // Blau
        }
        noStroke();
        let x = map(point.x, 0, 1, margins.left, width - margins.right);
        let y = map(point.y, 0, 1, height - margins.bottom, margins.top);
        ellipse(x, y, 8, 8);
    }
}

function drawUI() {
    let entropy = calculateEntropy(confidence);
    
    fill(0);
    textSize(16);
    textAlign(LEFT);
    text(`Klassifikationssicherheit: ${(confidence * 100).toFixed(1)}%`, 20, 580);
    text(`Entropie: ${entropy.toFixed(3)} bits`, 250, 580);
    
    textAlign(CENTER);
    text('Position', width/2, height - margins.bottom + 30);
    push();
    translate(20, height/2);
    rotate(-PI/2);
    text('Wahrscheinlichkeit', 0, 0);
    pop();
}

// p5.js Hauptfunktionen zuletzt
function setup() {
    let cnv = createCanvas(600, 600); // Erhöhen der Höhe um 10%
    cnv.parent('sketch-holder');
    
    slider = createSlider(50, 99, 50);
    slider.style('width', '250px');
    
    // Positionierung des Sliders relativ zum Canvas
    let canvasPos = cnv.position();
    slider.position(canvasPos.x + 175, canvasPos.y + height + -60); // Slider nach unten verschieben
    
    slider.input(() => {
        confidence = slider.value() / 100;
        generatePoints();
    });
    
    generatePoints();
}

function draw() {
    background(240);
    drawGrid();
    drawPoints();
    drawUI();
}
