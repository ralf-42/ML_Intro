# Diamonds Gradio App

Aus `b840_data_app_gradio_pipeline_diamonds.ipynb` extrahierte Gradio-App zur Schätzung von Diamantenpreisen.

## Voraussetzungen

- Docker Desktop oder eine andere Docker-Umgebung
- Freier lokaler Port `7860`
- Terminal im Ordner `docker/`
- Diese Dateien im Ordner:

```text
app.py
requirements.txt
Dockerfile
.dockerignore
KURS_DEPLOYMENT.md
REGISTRY_DEPLOYMENT.md
diamonds_pipeline.joblib
```

## Standardablauf

In den Docker-Ordner wechseln:

```bash
cd C:\Users\ralfb\OneDrive\Desktop\Kurse\ML_Intro\01_notebook\08_save_load_deploy\docker
```

Docker Image bauen:

```bash
docker build -t ml-intro-diamonds-gradio .
```

Image prüfen:

```bash
docker images ml-intro-diamonds-gradio
```

Container starten:

```bash
docker run --rm --name ml-intro-diamonds-gradio-live -p 7860:7860 ml-intro-diamonds-gradio
```

App im Browser öffnen:

```text
http://localhost:7860
```

In einem zweiten Terminal prüfen, ob der Container läuft:

```bash
docker ps
```

Erwartete Port-Ausgabe:

```text
0.0.0.0:7860->7860/tcp
```

## Build vs. Start

Der `docker build`-Schritt muss **nicht vor jedem Start** wiederholt werden:

- `docker build` erzeugt das Image `ml-intro-diamonds-gradio` und speichert es lokal in Docker (sichtbar über `docker images`).
- `docker run` startet lediglich einen neuen Container aus diesem bereits vorhandenen Image — kein erneuter Build nötig.
- Durch `--rm` wird nach dem Stoppen nur der **Container** entfernt, das **Image** bleibt erhalten.

Ein erneuter Build ist nur nötig, wenn sich `app.py`, `requirements.txt`, `Dockerfile` oder `diamonds_pipeline.joblib` geändert haben. Für jeden weiteren Start genügt der `docker run`-Befehl aus dem Abschnitt „Standardablauf".

## Container stoppen

Den laufenden Container stoppen:

```bash
docker stop ml-intro-diamonds-gradio-live
```

Bei Start mit `--rm` wird der Container danach automatisch entfernt.

## Fehlerfall: Port 7860 ist belegt

Typische Fehlermeldung:

```text
Bind for 0.0.0.0:7860 failed: port is already allocated
```

Laufende Container anzeigen:

```bash
docker ps
```

Wichtig bei der Ausgabe:

```text
0.0.0.0:7860->7860/tcp   Host-Port 7860 ist belegt
7860/tcp                 nur interner Container-Port, kein Host-Port-Mapping
```

Wenn `ml-intro-diamonds-gradio-live` den Port belegt:

```bash
docker stop ml-intro-diamonds-gradio-live
```

Wenn ein zufällig benannter Container den Port belegt, den angezeigten Namen stoppen, z. B.:

```bash
docker stop peaceful_ardinghelli
```

Danach den Container erneut starten:

```bash
docker run --rm --name ml-intro-diamonds-gradio-live -p 7860:7860 ml-intro-diamonds-gradio
```

## Fehlerfall: Container-Name ist bereits vergeben

Typische Fehlermeldung:

```text
The container name "/ml-intro-diamonds-gradio-live" is already in use
```

Alle Container mit Status anzeigen:

```bash
docker ps -a
```

Falls der Container noch läuft:

```bash
docker stop ml-intro-diamonds-gradio-live
```

Falls der Container bereits gestoppt ist, aber den Namen noch reserviert:

```bash
docker rm ml-intro-diamonds-gradio-live
```

Danach erneut starten:

```bash
docker run --rm --name ml-intro-diamonds-gradio-live -p 7860:7860 ml-intro-diamonds-gradio
```

## Alternative: anderen Host-Port verwenden

Wenn `7860` absichtlich belegt bleiben soll, kann die App auf einem anderen Host-Port gestartet werden:

```bash
docker run --rm --name ml-intro-diamonds-gradio-live -p 7861:7860 ml-intro-diamonds-gradio
```

Dann im Browser öffnen:

```text
http://localhost:7861
```

## Lokal ohne Docker starten

Optional kann die App auch direkt mit Python gestartet werden:

```bash
pip install -r requirements.txt
python app.py
```

## Deployment

### Variante 1: Docker-Ordner weitergeben

Den kompletten Ordner `docker/` an Dritte weitergeben. Die andere Person baut und startet das Image dann lokal:

```bash
docker build -t ml-intro-diamonds-gradio .
docker run --rm --name ml-intro-diamonds-gradio-live -p 7860:7860 ml-intro-diamonds-gradio
```

### Variante 2: Fertiges Image weitergeben

Image als Datei exportieren:

```bash
docker save -o ml-intro-diamonds-gradio.tar ml-intro-diamonds-gradio
```

Die andere Person importiert und startet das Image:

```bash
docker load -i ml-intro-diamonds-gradio.tar
docker run --rm --name ml-intro-diamonds-gradio-live -p 7860:7860 ml-intro-diamonds-gradio
```

### Variante 3: Registry verwenden

Für produktivere Nutzung kann das Image in eine Container Registry gepusht werden, z. B. Docker Hub, GitHub Container Registry oder eine interne Registry. Details stehen in `REGISTRY_DEPLOYMENT.md`.
