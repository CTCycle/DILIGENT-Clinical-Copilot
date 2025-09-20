# Pharmagent

## 1. Introduction
Pharmagent is a LLM-powered solution to perform Drug induced liver injury (DILI) analysis without human assistance. The app is developed using a FASTAPI for the backend and gradio to create a nice and intuitive UI.

It makes use of Ollama to interact with local models and keep the entire process compliant with high privacy standard and completely free, even though it is optionally possible to use online LLM services such as ChatGPT and Gemini as well.

## 2. Installation 
The installation process for Windows is fully automated. Simply run the script *start_on_windows.bat* to begin. During its initial execution, the script installs portable Python, necessary dependencies, minimizing user interaction and ensuring all components are ready for local use.  

## 3. How to use
On Windows, run *start_on_windows.bat* to launch the application. Please note that some antivirus software, such as Avast, may flag or quarantine python.exe when called by the .bat file. If you encounter unusual behavior, consider adding an exception in your antivirus settings.

The intuitive interface facilitates easy interactions with the Pharmagent core functionalities.

## 3.1 Setup and Maintenance
You can run *setup_and_maintenance.bat* to start the external tools for maintenance with the following options:

- **Update project:** check for updates from Github
- **Remove logs:** remove all logs file from *resources/logs*

### 3.1 Resources
This folder organizes dataset and tokenizers benchmark results. By default, all data is stored within an SQLite database. To visualize and interact with the SQLite database, we recommend downloading and installing the DB Browser for SQLite, available at: https://sqlitebrowser.org/dl/. The directory structure includes the following folders:

- **database:** tokenizers benchmark results will be stored centrally within the main database *TokenBenchy_database.db*. Graphical evaluation outputs for the performed benchmarks will be saved separately in *database/evaluation*. Moreover, this folder contains the downloaded datasets that are used to test the tokenizers performance (open access datasets are saved in *datasets/open* while the custom dataset is saved into *datasets/custom*). Last but not least, the downloaded tokenizers are saved in *database/tokenizers* following the same organisation of the datasets folder. 

- **logs:** log files are saved here

- **templates:** reference template files can be found here

**Environmental variables** are stored in the *app* folder (within the project folder). For security reasons, this file is typically not uploaded to GitHub. Instead, you must create this file manually by copying the template from *resources/templates/.env* and placing it in the *app* directory.

| Variable              | Description                                              |
|-----------------------|----------------------------------------------------------|
| ACCESS_TOKEN          | HuggingFace access token (required for some tokenizers)  |
| TF_CPP_MIN_LOG_LEVEL  | TensorFlow logging verbosity                             |
| MPLBACKEND            | Matplotlib backend, keep default as Agg                  |


## 4. License
This project is licensed under the terms of the MIT license. See the LICENSE file for details.

