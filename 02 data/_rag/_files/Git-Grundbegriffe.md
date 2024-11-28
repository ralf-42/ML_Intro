
Git ist ein weit verbreitetes Versionskontrollsystem, das es Entwicklern ermöglicht, effizient an Projekten zu arbeiten, indem es die Änderungen am Code über die Zeit hinweg verfolgt. Hier sind einige der Grundbegriffe, die für die Nutzung von Git wesentlich sind:

### Repository (Repo)

Ein Repository ist ein Speicherort für dein Projekt. Es enthält alle Dateien des Projekts sowie die Versionsgeschichte aller Dateien. Ein Repository kann lokal auf deinem Computer gespeichert sein oder auf einem Server wie GitHub, GitLab oder Bitbucket gehostet werden. Es dient als Basis für die Zusammenarbeit in Projekten und ermöglicht es, Änderungen nachzuvollziehen.

### Branch

Ein Branch in Git ermöglicht es dir, von der Hauptlinie der Entwicklung abzuzweigen und parallel daran zu arbeiten, ohne die Hauptlinie zu beeinflussen. Dies ist besonders nützlich, wenn du an neuen Features oder Bugfixes arbeitest. Der Standardbranch in Git heißt `master` oder `main`, aber du kannst so viele Branches erstellen, wie du für deine Arbeit benötigst. Nachdem du in einem Branch fertig gearbeitet hast, kannst du ihn in den Hauptbranch zurückführen (merge).

### Commit

Ein Commit ist eine Aufzeichnung von Änderungen, die an einem oder mehreren Dateien vorgenommen wurden. Jeder Commit speichert einen Zustand deines Projekts, den du später wiederherstellen kannst. Commits enthalten auch eine Commit-Nachricht, die eine Beschreibung der vorgenommenen Änderungen bietet, sowie Informationen über den Autor des Commits und einen Zeitstempel.

### Push

Mit dem Befehl `git push` werden lokale Änderungen (Commits) zu einem entfernten Repository hochgeladen. Dies ermöglicht es anderen, Zugriff auf deine Änderungen zu haben oder mit diesen zu arbeiten. Wenn du in einem Team arbeitest, ist es üblich, deine Änderungen regelmäßig zu pushen, damit das Team auf dem aktuellen Stand bleibt.

### Pull

Der Befehl `git pull` aktualisiert dein lokales Repository mit den neuesten Änderungen aus dem entfernten Repository. Dies beinhaltet das Herunterladen (fetch) der neuesten Änderungen und das Zusammenführen (merge) dieser Änderungen in deinen aktuellen Arbeitsbranch. `git pull` wird oft verwendet, um sicherzustellen, dass dein lokales Repository aktuell ist, bevor neue Änderungen vorgenommen werden.

### Merge

Ein Merge ist der Prozess, bei dem Änderungen aus einem Branch in einen anderen übertragen werden. Dies wird typischerweise verwendet, um einen Entwicklungsbranch zurück in den Hauptbranch (`master` oder `main`) zu integrieren, nachdem die Arbeit in dem Entwicklungszweig abgeschlossen ist. Merges können automatisch durchgeführt werden, wenn es keine konkurrierenden Änderungen gibt, oder sie können manuelle Konfliktlösung erfordern, wenn die gleichen Teile des Codes in beiden Branches geändert wurden.