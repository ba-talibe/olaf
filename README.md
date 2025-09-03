# Ontology-learning

Since the beginning of the century, research on ontology learning has gained popularity. Automatically extracting and structuring knowledge relevant to a domain of interest from unstructured data is a major scientific challenge. We propose a new approach with modular ontology learning framework considering tasks from data pre-processing to axiom extraction. Whereas previous contributions considered ontology learning systems as tools to help the domain expert, we developed the proposed framework with full automation in mind.

Resources:

- The documentation is available here: [OLAF](https://wikit-ai.github.io/olaf/index.html)
- [Poster](./docs/Poster_OLAF_2023.pdf)
- Our research paper has been published at [KES 2023](http://kes2023.kesinternational.org/).

## Installation

For usage :

```
pip install git+https://github.com/wikit-ai/olaf

```

For contribution :

```
git clone https://github.com/wikit-ai/olaf.git
cd olaf
python3 -m venv ./venv
source venv/bin/activate
pip install .
```

# Installation de Java et du fichier `robot.jar`

Ce guide explique comment installer Java et télécharger le fichier `robot.jar`, un outil indispensable pour effectuer un raisonnement sur une ontologie.

## Prérequis

- Un système d'exploitation compatible (Linux, macOS ou Windows).
- Une connexion Internet active.

## Étape 1 : Installer Java

`robot.jar` nécessite Java pour fonctionner. Suivez les étapes ci-dessous pour installer Java :

### Sous Linux

1. Ouvrez un terminal.
2. Mettez à jour la liste des paquets :
   ```bash
   sudo apt update
   ```
3. Installez OpenJDK (version 11 ou supérieure) :
   ```bash
   sudo apt install openjdk-11-jdk
   ```
4. Vérifiez l'installation de Java :
   ```bash
   java -version
   ```
   Vous devriez voir une sortie indiquant la version de Java installée.

### Sous macOS

1. Installez [Homebrew](https://brew.sh/) si ce n'est pas déjà fait.
2. Installez OpenJDK :
   ```bash
   brew install openjdk
   ```
3. Configurez les variables d'environnement si nécessaire (suivez les instructions affichées après l'installation).
4. Vérifiez l'installation de Java :
   ```bash
   java -version
   ```

### Sous Windows

1. Téléchargez l'installateur de Java depuis le site officiel d'Oracle ou utilisez OpenJDK via [AdoptOpenJDK](https://adoptopenjdk.net/).
2. Suivez les instructions de l'installateur.
3. Vérifiez l'installation de Java en ouvrant une invite de commande et en exécutant :
   ```cmd
   java -version
   ```

## Étape 2 : Télécharger le fichier `robot.jar`

1. Accédez au lien suivant pour télécharger `robot.jar` :
   [Télécharger robot.jar](https://github.com/ontodev/robot/releases/download/v1.9.8/robot.jar)
2. Enregistrez le fichier dans un répertoire de votre choix (par exemple, `/home/utilisateur/robot/` sous Linux).

## Étape 3 : Utiliser `robot.jar`

1. Ouvrez un terminal ou une invite de commande.
2. Naviguez jusqu'au répertoire contenant `robot.jar` :
   ```bash
   cd /chemin/vers/le/dossier/robot
   ```
3. Exécutez une commande avec `robot.jar` pour vérifier son fonctionnement :
   ```bash
   java -jar robot.jar --help
   ```
   Vous devriez voir une liste des commandes disponibles.

## Configuration des variables d'environnement

Pour utiliser `robot.jar`, vous devez configurer les variables d'environnement suivantes dans un fichier `.env` :

```properties
JAVA_EXE=/chemin/vers/votre/java/java.exe
ROBOT_JAR=/chemin/vers/votre/robot.jar
```

- **JAVA_EXE** : Chemin vers l'exécutable Java installé sur votre système.
- **ROBOT_JAR** : Chemin vers le fichier `robot.jar` téléchargé.

Assurez-vous que ces chemins sont corrects et pointent vers les fichiers appropriés sur votre système.

## Ressources supplémentaires

- Documentation officielle de ROBOT : [https://robot.obolibrary.org/](https://robot.obolibrary.org/)
- Dépôt GitHub de ROBOT : [https://github.com/ontodev/robot](https://github.com/ontodev/robot)

---

Si vous rencontrez des problèmes, n'hésitez pas à consulter la documentation officielle ou à ouvrir une issue sur le dépôt GitHub de ROBOT.

## Quick-start

Pipelines can be run with the following command: `olaf run demo_pipeline`.
Pipeline components are displayed with the following command: `olaf show demo_pipeline`.
The text used can be updated in the file `data/demo.txt`.

An example on how the library can be used is available in `demontrators/demo_test.ipynb`.

One example of OLAF usage for LLM components evaluation is also available here : [https://github.com/wikit-ai/olaf-llm-eswc2024](https://github.com/wikit-ai/olaf-llm-eswc2024).

## How to contribute

When an algorithm is missing you can contribute by adding it. Please refer to the [developer note](./docs/dev_notes.md) in the documentation for more detailed information.

## Citing us

> Marion Schaeffer, Matthias Sesboüé, Jean-Philippe Kotowicz, Nicolas Delestre, Cecilia Zanni-Merk,
> OLAF: An Ontology Learning Applied Framework,
> Procedia Computer Science,
> Volume 225,
> 2023,
> Pages 2106-2115,
> ISSN 1877-0509,
> https://doi.org/10.1016/j.procs.2023.10.201.
> (https://www.sciencedirect.com/science/article/pii/S1877050923013595)

## License

This project is licensed under the Apache-2.0 License.
