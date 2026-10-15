// noprotect

let originalSketch;
let filteredSketch;

const filters = {
    horizontalEdge: {
        name: "Horizontale Kanten",
        kernel: [[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]],
        description: "Erkennt horizontale Kanten durch vertikale Gradienten."
    },
    verticalEdge: {
        name: "Vertikale Kanten",
        kernel: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        description: "Erkennt vertikale Kanten durch horizontale Gradienten."
    },
    edgeDetection: {
        name: "Kantenerkennung",
        kernel: [[-1, -1, -1], [-1,  8, -1], [-1, -1, -1]],
        description: "Erkennt Kanten durch Hervorhebung starker Helligkeitsunterschiede."
    },
    sharpen: {
        name: "Schärfen",
        kernel: [[ 0, -1,  0], [-1,  5, -1], [ 0, -1,  0]],
        description: "Verstärkt Kontraste und Details im Bild."
    },
    gaussianBlur: {
        name: "Weichzeichnen",
        kernel: [[1/16, 2/16, 1/16], [2/16, 4/16, 2/16], [1/16, 2/16, 1/16]],
        description: "Weichzeichnung durch gewichtete Mittelwertbildung."
    },
    emboss: {
        name: "Relief",
        kernel: [[-2, -1,  0], [-1,  1,  1], [ 0,  1,  2]],
        description: "Erzeugt einen 3D-Relief-Effekt."
    },
    sobelX: {
        name: "Sobel X",
        kernel: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        description: "Erkennt vertikale Kanten im Bild."
    }
};

// Original Sketch
originalSketch = new p5((p) => {
    let img = null;
    
    p.setup = function() {
        const canvas = p.createCanvas(400, 400);
        canvas.parent('originalCanvas');
        canvas.style('margin', 'auto');
        p.pixelDensity(1);
        p.background(240);
        p.noLoop();
        p.imageMode(p.CENTER);

        const uploadInput = document.getElementById('imageUpload');
        if (uploadInput) {
            uploadInput.addEventListener('change', (event) => {
                const file = event.target.files[0];
                if (file && file.type.startsWith('image')) {
                    const fileURL = URL.createObjectURL(file);
                    p.loadAndDisplayImage(fileURL);
                }
            });
        }
    };
    
    p.loadAndDisplayImage = function(file) {
        p.loadImage(file, (loadedImg) => {
            img = loadedImg;
            
            const scale = Math.min(p.width / img.width, p.height / img.height);
            const newWidth = Math.floor(img.width * scale);
            const newHeight = Math.floor(img.height * scale);
            img.resize(newWidth, newHeight);
            
            p.background(240);
            p.image(img, p.width / 2, p.height / 2);
            
            setTimeout(() => {
                if (filteredSketch && typeof filteredSketch.applyFilterToImage === 'function') {
                    filteredSketch.applyFilterToImage(img);
                }
            }, 50);
        });
    };
    
    p.getImage = function() {
        return img;
    };
}, 'originalCanvas');

// Filtered Sketch
filteredSketch = new p5((p) => {
    let filteredImg = null;
    
    p.setup = function() {
        const canvas = p.createCanvas(400, 400);
        canvas.parent('filteredCanvas');
        canvas.style('margin', 'auto');
        p.pixelDensity(1);
        p.background(240);
        p.noLoop();
        p.imageMode(p.CENTER);

        const filterSelect = document.getElementById('filterSelect');
        if (filterSelect) {
            filterSelect.addEventListener('change', (event) => {
                const filterType = event.target.value;
                const filter = filters[filterType];
                
                if (document.getElementById('filterName')) {
                    document.getElementById('filterName').textContent = filter.name;
                }
                if (document.getElementById('filterDescription')) {
                    document.getElementById('filterDescription').textContent = filter.description;
                }
                
                if (originalSketch) {
                    const originalImg = originalSketch.getImage();
                    if (originalImg) {
                        p.applyFilterToImage(originalImg);
                    }
                }
            });
        }
    };
    
    p.applyFilterToImage = function(sourceImg) {
        if (!sourceImg) return;
        
        const filterSelectElement = document.getElementById('filterSelect');
        if (!filterSelectElement) return;
        
        const filterType = filterSelectElement.value;
        const k = filters[filterType].kernel; // Abkürzung für den Kernel
        
        filteredImg = p.createImage(sourceImg.width, sourceImg.height);
        filteredImg.copy(sourceImg, 0, 0, sourceImg.width, sourceImg.height, 0, 0, sourceImg.width, sourceImg.height);
        
        filteredImg.loadPixels();
        sourceImg.loadPixels();
        
        const w = filteredImg.width;
        const h = filteredImg.height;
        const src = sourceImg.pixels;
        const out = filteredImg.pixels;
        
        // Loop ohne innere Kernel-Schleifen (Loop Unrolling)
        for (let y = 1; y < h - 1; y++) {
            for (let x = 1; x < w - 1; x++) {
                
                // Indizes für alle 9 Nachbarpixel im 1D-Array vorbereiten
                const idxN00 = ((y - 1) * w + (x - 1)) * 4;
                const idxN01 = ((y - 1) * w + x) * 4;
                const idxN02 = ((y - 1) * w + (x + 1)) * 4;
                
                const idxN10 = (y * w + (x - 1)) * 4;
                const idxN11 = (y * w + x) * 4;
                const idxN12 = (y * w + (x + 1)) * 4;
                
                const idxN20 = ((y + 1) * w + (x - 1)) * 4;
                const idxN21 = ((y + 1) * w + x) * 4;
                const idxN22 = ((y + 1) * w + (x + 1)) * 4;
                
                // Rot-Kanal falten
                let sumR = src[idxN00] * k[0][0] + src[idxN01] * k[0][1] + src[idxN02] * k[0][2] +
                           src[idxN10] * k[1][0] + src[idxN11] * k[1][1] + src[idxN12] * k[1][2] +
                           src[idxN20] * k[2][0] + src[idxN21] * k[2][1] + src[idxN22] * k[2][2];
                           
                // Grün-Kanal falten
                let sumG = src[idxN00+1] * k[0][0] + src[idxN01+1] * k[0][1] + src[idxN02+1] * k[0][2] +
                           src[idxN10+1] * k[1][0] + src[idxN11+1] * k[1][1] + src[idxN12+1] * k[1][2] +
                           src[idxN20+1] * k[2][0] + src[idxN21+1] * k[2][1] + src[idxN22+1] * k[2][2];
                           
                // Blau-Kanal falten
                let sumB = src[idxN00+2] * k[0][0] + src[idxN01+2] * k[0][1] + src[idxN02+2] * k[0][2] +
                           src[idxN10+2] * k[1][0] + src[idxN11+2] * k[1][1] + src[idxN12+2] * k[1][2] +
                           src[idxN20+2] * k[2][0] + src[idxN21+2] * k[2][1] + src[idxN22+2] * k[2][2];

                // Ziel-Index beschreiben
                const outIdx = (y * w + x) * 4;
                out[outIdx]     = sumR < 0 ? 0 : (sumR > 255 ? 255 : sumR);
                out[outIdx + 1] = sumG < 0 ? 0 : (sumG > 255 ? 255 : sumG);
                out[outIdx + 2] = sumB < 0 ? 0 : (sumB > 255 ? 255 : sumB);
                out[outIdx + 3] = 255;
            }
        }
        
        filteredImg.updatePixels();
        
        p.background(240);
        p.image(filteredImg, p.width / 2, p.height / 2);
    };
}, 'filteredCanvas');