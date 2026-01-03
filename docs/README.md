# GitHub Pages Dokumentation - ML_Intro

Diese Dokumentation wird über GitHub Pages veröffentlicht.

## GitHub Pages aktivieren

1. Gehe zu deinem GitHub Repository: `https://github.com/ralf-42/ML_Intro`
2. Klicke auf **Settings** (Einstellungen)
3. Wähle im linken Menü **Pages** aus
4. Unter **Source** (Quelle):
   - Branch: `main` (oder `master`)
   - Folder: `/docs`
5. Klicke auf **Save**

Nach 2-3 Minuten ist die Seite verfügbar unter:
`https://ralf-42.github.io/ML_Intro/`

## Struktur

```
docs/
├── _config.yml              # Jekyll-Konfiguration
├── _includes/               # HTML-Includes (head_custom.html)
├── _sass/custom/            # Custom SCSS-Styles
├── assets/js/               # JavaScript (callouts.js)
├── concepts/                # Konzepte (leer, für zukünftige Inhalte)
├── deployment/              # Deployment-Guides (leer)
├── frameworks/              # Framework-Guides (leer)
├── legal/                   # Impressum, Datenschutz, Haftungsausschluss
├── projekte/                # Projektbeispiele (leer)
├── regulatorisches/         # Ethik & Recht (leer)
├── ressourcen/              # Ressourcen & Tools (leer)
├── index.md                 # Startseite
└── *.md                     # Navigationsseiten
```

## Theme

Verwendet wird das **Just the Docs** Theme mit folgenden Features:
- ✅ Responsive Design
- ✅ Suche
- ✅ Inhaltsverzeichnis (TOC)
- ✅ Mermaid-Diagramme
- ✅ Custom Callouts (Obsidian-kompatibel)
- ✅ Syntax-Highlighting

## Callouts

Markdown-Callouts (Obsidian-Syntax) werden automatisch transformiert:

```markdown
> [!NOTE]
> Dies ist ein Hinweis
```

Unterstützte Typen: NOTE, INFO, TIP, WARNING, DANGER, EXAMPLE, SUCCESS, etc.

## Anpassungen

### Schriftgröße ändern
Siehe: `_sass/custom/FONT_SIZE_GUIDE.md`

### Farben & Layout
Siehe: `_sass/custom/custom.scss`

### Navigation
Navigation wird automatisch aus den `nav_order` Werten in den Markdown-Dateien generiert.

## Lokale Vorschau

```bash
# Jekyll lokal installieren (einmalig)
gem install bundler jekyll

# Im docs/ Verzeichnis
bundle install
bundle exec jekyll serve

# Öffne: http://localhost:4000/ML_Intro/
```

## Weitere Informationen

- [Just the Docs Dokumentation](https://just-the-docs.com/)
- [GitHub Pages Dokumentation](https://docs.github.com/en/pages)
- [Jekyll Dokumentation](https://jekyllrb.com/docs/)
