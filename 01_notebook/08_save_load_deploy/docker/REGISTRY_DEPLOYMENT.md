# Deployment über Container Registries

Diese Datei beschreibt kurz zwei Wege, um das Docker Image anderen Personen bereitzustellen: GitHub Container Registry und eine einfache interne Registry.

## GitHub Container Registry

Voraussetzungen:

- GitHub-Account
- GitHub Personal Access Token mit `read:packages` und `write:packages`
- Docker Desktop oder eine andere Docker-Umgebung

Bei GitHub Container Registry anmelden:

```bash
docker login ghcr.io -u DEIN_GITHUB_USERNAME
```

Image taggen:

```bash
docker tag ml-intro-diamonds-gradio ghcr.io/DEIN_GITHUB_USERNAME/ml-intro-diamonds-gradio:latest
```

Image hochladen:

```bash
docker push ghcr.io/DEIN_GITHUB_USERNAME/ml-intro-diamonds-gradio:latest
```

Image auf einem anderen Rechner starten:

```bash
docker pull ghcr.io/DEIN_GITHUB_USERNAME/ml-intro-diamonds-gradio:latest
docker run --rm -p 7860:7860 ghcr.io/DEIN_GITHUB_USERNAME/ml-intro-diamonds-gradio:latest
```

## Einfache interne Registry

Eine lokale Registry als Container starten:

```bash
docker run -d -p 5000:5000 --name registry registry:2
```

Image taggen:

```bash
docker tag ml-intro-diamonds-gradio localhost:5000/ml-intro-diamonds-gradio:latest
```

Image in die Registry hochladen:

```bash
docker push localhost:5000/ml-intro-diamonds-gradio:latest
```

Image aus der Registry starten:

```bash
docker pull localhost:5000/ml-intro-diamonds-gradio:latest
docker run --rm -p 7860:7860 localhost:5000/ml-intro-diamonds-gradio:latest
```

## Empfehlung für den Kurs

Für einzelne Rechner oder Übungen reicht der komplette `docker/`-Ordner. Für die Weitergabe an mehrere Personen ist GitHub Container Registry meist einfacher. Eine interne Registry ist sinnvoll, wenn Images innerhalb eines Unternehmens oder Schulungsnetzwerks bleiben sollen.
