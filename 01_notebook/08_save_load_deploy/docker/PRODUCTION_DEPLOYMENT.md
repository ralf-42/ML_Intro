# Produktionsnaher Betrieb

Der Ablauf in `KURS_DEPLOYMENT.md` ist für Kurs, Demo und lokale Weitergabe gedacht. Für einen echten Produktionsbetrieb sind zusätzliche Maßnahmen nötig.

Registry-spezifische Schritte wie `docker login`, `docker tag`, `docker push` und `docker pull` stehen gesammelt in `REGISTRY_DEPLOYMENT.md`. Dieses Dokument konzentriert sich auf den Betrieb der Anwendung.

## Zielbild

Im Produktionsbetrieb wird das Image nicht auf dem Zielsystem gebaut, sondern aus einer Container Registry bezogen. Der Container läuft im Hintergrund, startet nach Neustarts automatisch wieder und wird über Logs, Monitoring und Healthchecks überwacht.

## Image versionieren

Für Produktion keine reinen `latest`-Deployments verwenden. Stattdessen einen festen Versions-Tag vergeben, z. B.:

```text
ml-intro-diamonds-gradio:1.0.0
```

Das Bauen, Taggen und Veröffentlichen des Images ist in `REGISTRY_DEPLOYMENT.md` beschrieben.

## Container produktionsnäher starten

Für einen einfachen produktionsnahen Betrieb mit Docker:

```bash
docker run -d \
  --name ml-intro-diamonds-gradio \
  --restart unless-stopped \
  -p 7860:7860 \
  ghcr.io/DEIN_GITHUB_USERNAME/ml-intro-diamonds-gradio:1.0.0
```

Unterschied zum Kursbetrieb:

- `-d` startet den Container im Hintergrund.
- `--restart unless-stopped` startet den Container nach Neustarts automatisch wieder.
- Ein fester Versions-Tag ersetzt `latest`.
- Das Image kommt aus einer Registry statt aus einem lokalen Build.
- Der Container wird ohne `--rm` gestartet, damit Logs und Status kontrolliert geprüft werden können.

## Logs prüfen

Einmalige Log-Ausgabe:

```bash
docker logs ml-intro-diamonds-gradio
```

Live-Logs anzeigen:

```bash
docker logs -f ml-intro-diamonds-gradio
```

## Container aktualisieren

Eine neue Version wird zuerst gebaut und in die Registry veröffentlicht. Die Details stehen in `REGISTRY_DEPLOYMENT.md`.

Auf dem Zielsystem wird anschließend der alte Container ersetzt:

```bash
docker stop ml-intro-diamonds-gradio
docker rm ml-intro-diamonds-gradio
docker run -d \
  --name ml-intro-diamonds-gradio \
  --restart unless-stopped \
  -p 7860:7860 \
  ghcr.io/DEIN_GITHUB_USERNAME/ml-intro-diamonds-gradio:1.0.1
```

Falls das neue Image auf dem Zielsystem noch nicht vorhanden ist, vorher ziehen:

```bash
docker pull ghcr.io/DEIN_GITHUB_USERNAME/ml-intro-diamonds-gradio:1.0.1
```

## HTTPS und Reverse Proxy

Für öffentlichen Zugriff sollte die App nicht direkt ungeschützt über Port `7860` veröffentlicht werden. Üblich ist ein Reverse Proxy wie Nginx, Traefik oder Caddy:

- HTTPS/TLS terminieren
- Domain auf die App routen
- Zugriff optional absichern
- Logs und Limits zentral behandeln

Beispiel-Ziel:

```text
https://diamonds.example.com
```

leitet intern weiter auf:

```text
http://localhost:7860
```

## Healthcheck

Für produktionsnahe Deployments sollte regelmäßig geprüft werden, ob die App erreichbar ist:

```bash
curl http://localhost:7860
```

In produktiveren Umgebungen wird der Healthcheck vom Deployment-System ausgeführt, z. B. Docker Compose, Kubernetes, Azure Container Apps, AWS ECS oder Cloud Run.

## Sicherheit

Für echten Produktionsbetrieb zusätzlich prüfen:

- Container möglichst nicht als `root` ausführen
- Abhängigkeiten regelmäßig aktualisieren
- Images regelmäßig neu bauen
- Keine Secrets im Image speichern
- Zugriff auf die App absichern, falls sie öffentlich erreichbar ist
- Modellartefakte versionieren und nachvollziehbar ablegen

## Empfehlung

Für den Kurs reicht der Ablauf aus `KURS_DEPLOYMENT.md`. Für produktionsnähere Demos sollte mindestens verwendet werden:

- versionierter Image-Tag
- Registry statt lokalem Build auf dem Zielsystem
- `docker run -d`
- `--restart unless-stopped`
- zentrale Logs
- Reverse Proxy mit HTTPS bei öffentlichem Zugriff

Die Registry-Schritte bleiben in `REGISTRY_DEPLOYMENT.md`; die Betriebsaspekte stehen hier.
