# Solar Analysis Project

This project is designed to analyze solar cell data, process photovoltaic parameters, and generate reports using Python. It utilizes a virtual environment to manage dependencies, ensuring a consistent setup across different systems.

## Features
**Data Analysis**: Analyze photovoltaic measurement data from solar cells.
**Report Generation**: Create PDF reports with results and graphs.
**Virtual Environment**: Ensures proper package installation without system-wide impact.

## Requirements
**Python**: Python 3.6 or higher
**Operating System**: Windows or macOS
**Dependencies**: Listed in requirements.txt

## Installation
### 1. Clone the Repository
First, clone the project repository to your local machine. Open a terminal and run:

```bash
git https://github.com/Smart-Energy-Materials-Group-UTU/Solar-Analysis-App.git

cd Solar-Analysis-App
```

### 2. Set Up the Virtual Environment
The project uses a virtual environment to manage dependencies. The provided scripts will automatically set up the environment for you.

#### **Windows**:

- Double-click the launch.bat file to set up the virtual environment and run the application.

- If the virtual environment does not exist, the script will create it and install the dependencies listed in requirements.txt.

#### **macOS/Linux**:

1. Open a terminal and navigate to the project directory.

2. Run the launch.sh script to set up the virtual environment and run the application.

```bash
chmod +x launch.sh  # Make the script executable
./launch.sh  # Run the script
```

The script will:
- Check if the virtual environment exists.
- Install dependencies from requirements.txt.
- Run the main application script main_algorithm.py.

## Dependencies
The project requires the following Python packages, which will be installed automatically using the requirements.txt file:
- `numpy`
- `matplotlib`
- `pandas`
- `scipy`
- `openpyxl`
- `seaborn`
- And many more as specified in requirements.txt.

## Usage
After running the appropriate launch script (launch.bat for Windows or launch.sh for macOS/Linux), the application will:
- Activate the virtual environment.
- Launch the main algorithm script (main_algorithm.py).

## Deactivating the Virtual Environment
Once the analysis is complete, the virtual environment will automatically be deactivated.

## Virtual Environment Issues
If you encounter issues with the virtual environment, ensure you have Python 3.6 or higher installed on your system. You can check your Python version by running:

```bash
python --version  # or python3 --version on macOS/Linux
```
If the virtual environment setup fails, you can create it manually by running:

```bash
# Windows (in PowerShell or Command Prompt)
python -m venv thesis

# macOS/Linux
python3 -m venv thesis
```
Then activate the environment manually:

Windows: thesis\Scripts\activate

macOS/Linux: source thesis/bin/activate

## Package Installation Issues
If you encounter issues installing dependencies, ensure that your pip is up to date. You can upgrade pip by running:

```bash
# Windows or macOS/Linux
python -m pip install --upgrade pip
```
Then, try installing the dependencies again:

```bash
pip install -r requirements.txt
```
## Contributing
If you'd like to contribute to the project, feel free to fork the repository, make changes, and submit a pull request.

## License and Branding Notice

This project is licensed under the GNU General Public License v3.0.
However, all branding elements—such as the application name, logos, and visual styling in the `Images/` folder—are protected and may not be modified or reused without explicit permission from the author.
Please respect the integrity of the original branding when redistributing or modifying this software.
