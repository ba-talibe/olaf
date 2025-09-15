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

# Java and `robot.jar` Installation

This guide explains how to install Java and download the `robot.jar` file, an essential tool for reasoning over an ontology.

## Prerequisites

- A compatible operating system (Linux, macOS, or Windows).
- An active Internet connection.

## Step 1: Install Java

`robot.jar` requires Java to run. Follow the steps below to install Java:

### On Linux

1. Open a terminal.
2. Update the package list:
   ```bash
   sudo apt update
   ```
3. Install OpenJDK (version 11 or higher):
   ```bash
   sudo apt install openjdk-11-jdk
   ```
4. Check the Java installation:
   ```bash
   java -version
   ```
   You should see output indicating the installed Java version.

### On macOS

1. Install [Homebrew](https://brew.sh/) if not already installed.
2. Install OpenJDK:
   ```bash
   brew install openjdk
   ```
3. Set environment variables if needed (follow the instructions displayed after installation).
4. Check the Java installation:
   ```bash
   java -version
   ```

### On Windows

1. Download the Java installer from the official Oracle website or use OpenJDK via [AdoptOpenJDK](https://adoptopenjdk.net/).
2. Follow the installer instructions.
3. Check the Java installation by opening a command prompt and running:
   ```cmd
   java -version
   ```

## Step 2: Download the `robot.jar` file

1. Go to the following link to download `robot.jar`:
   [Download robot.jar](https://github.com/ontodev/robot/releases/download/v1.9.8/robot.jar)
2. Save the file in a directory of your choice (e.g., `/home/user/robot/` on Linux).

## Step 3: Use `robot.jar`

1. Open a terminal or command prompt.
2. Navigate to the directory containing `robot.jar`:
   ```bash
   cd /path/to/robot/folder
   ```
3. Run a command with `robot.jar` to check it works:
   ```bash
   java -jar robot.jar --help
   ```
   You should see a list of available commands.

## Environment Variable Configuration

To use `robot.jar`, you must configure the following environment variables in a `.env` file:

```properties
JAVA_EXE=/path/to/your/java/java.exe
ROBOT_JAR=/path/to/your/robot.jar
```

- **JAVA_EXE**: Path to the Java executable installed on your system.
- **ROBOT_JAR**: Path to the downloaded `robot.jar` file.

Make sure these paths are correct and point to the appropriate files on your system.

## Additional Resources

- Official ROBOT documentation: [https://robot.obolibrary.org/](https://robot.obolibrary.org/)
- ROBOT GitHub repository: [https://github.com/ontodev/robot](https://github.com/ontodev/robot)

---

If you encounter any issues, please refer to the official documentation or open an issue on the ROBOT GitHub repository.

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
