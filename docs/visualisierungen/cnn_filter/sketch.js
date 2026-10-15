let currentFilter = 0;
const canvasSize = 400;
const gridSize = 20;

// Filter-Definitionen
const filters = [
    {
        name: 'Kantenerkennung (horizontal)',
        matrix: [
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1]
        ],
        description: 'Erkennt horizontale Kanten im Bild durch Differenzbildung zwischen oberen und unteren Pixeln.'
    },
    {
        name: 'Kantenerkennung (vertikal)',
        matrix: [
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ],
        description: 'Erkennt vertikale Kanten durch Differenzbildung zwischen linken und rechten Pixeln.'
    },
    {
        name: 'Schärfung',
        matrix: [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ],
        description: 'Verstärkt Zentral-Pixel und reduziert Nachbarpixel, um Details hervorzuheben.'
    },
    {
        name: 'Weichzeichnung (Blur)',
        matrix: [
            [1/9, 1/9, 1/9],
            [1/9, 1/9, 1/9],
            [1/9, 1/9, 1/9]
        ],
        description: 'Mittelt benachbarte Pixel für einen Weichzeichnungseffekt.'
    },
    {
        name: 'Sobel-Operator',
        matrix: [
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ],
        description: 'Fortgeschrittener Kantenerkennungsfilter mit Gewichtung der zentralen Pixel.'
    }
];

function setup() {
    // Canvas erstellen und im container platzieren
    const canvas = createCanvas(canvasSize, canvasSize);
    canvas.parent('canvas-container');
    
    // Buttons mit Event-Listenern versehen
    for (let i = 0; i < filters.length; i++) {
        const button = document.getElementById(`filter${i}`);
        button.addEventListener('click', () => {
            currentFilter = i;
            updateUI();
            // Matrix wird automatisch im draw() neu gezeichnet
        });
    }
    
    // Initial UI update
    updateUI();
    
    // Pixeldichte auf 1 setzen für bessere Performance
    pixelDensity(1);
}

function draw() {
    // Zeichne weißen Hintergrund
    background(255);
    
    // Zeichne die Filtermatrix
    const matrixSize = 180;  // Größe der gesamten Matrix
    const cellSize = matrixSize / 3;  // Größe einer einzelnen Zelle
    const startX = (width - matrixSize) / 2;  // Zentriere horizontal
    const startY = (height - matrixSize) / 2;  // Zentriere vertikal
    
    const matrix = filters[currentFilter].matrix;
    
    for(let i = 0; i < 3; i++) {
        for(let j = 0; j < 3; j++) {
            const value = matrix[i][j];
            const x = startX + j * cellSize;
            const y = startY + i * cellSize;
            
            // Zeichne Zellenhintergrund
            noStroke();
            if(value > 0) {
                fill(0, 0, 255, value * 50);
            } else if(value < 0) {
                fill(255, 0, 0, abs(value) * 50);
            } else {
                fill(240);
            }
            rect(x, y, cellSize, cellSize);
            
            // Zeichne Zellenwert
            textAlign(CENTER, CENTER);
            textSize(16);
            fill(0);
            const displayValue = typeof value === 'number' ? 
                value.toFixed(2).replace(/\.?0+$/, '') : value;
            text(displayValue, x + cellSize/2, y + cellSize/2);
            
            // Zeichne Zellenrahmen
            stroke(200);
            noFill();
            rect(x, y, cellSize, cellSize);
        }
    }
}

function mouseDragged() {
    // Diese Funktion wird nicht mehr benötigt, da wir keine interaktive
    // Zeichenfläche mehr haben
    return false;
}

function updateUI() {
    // Buttons aktualisieren
    for (let i = 0; i < filters.length; i++) {
        const button = document.getElementById(`filter${i}`);
        if (i === currentFilter) {
            button.classList.add('active');
        } else {
            button.classList.remove('active');
        }
    }
    
    // Beschreibung aktualisieren
    document.getElementById('description').textContent = 
        filters[currentFilter].description;
}