# CS 5330 Labs and Activity Learnings

## How to run the app
You can run the app using two methods: using Poetry or using the system's Python interpreter.
Make sure you have **Python 3.12** or higher installed on your system.

### Using Poetry (Recommended)
1. If you haven't installed Poetry yet, you can do so by following the instructions [here](https://python-poetry.org/docs/). Optionally, you can configure Poetry to create virtual environments within the project directory. This is recommended for better project isolation. Run the following command:
```shell
poetry config virtualenvs.in-project true
```

2. Install the dependencies
```shell
poetry install
```

3. Run the main file
```shell
poetry run python src/main.py
```

### Using the system's Python interpreter
There are two ways to run the app using the system's Python interpreter: using global interpreter or creating a virtual environment.

#### Using global interpreter
1. If you want to use the system's Python interpreter then you need install the dependencies from the `requirements.txt` file using pip. Run the following command:
```shell
pip install -r requirements.txt
```

2. Run the main file
```shell
python src/main.py
```

#### Using a virtual environment
1. Create a virtual environment
```shell
python -m venv .venv
```

2. Activate the virtual environment
```shell
# On Windows
.venv\Scripts\activate.bat

# On Unix or MacOS
source .venv/bin/activate
```

3. Install the dependencies
```shell
pip install -r requirements.txt
```

4. Run the main file
```shell
python src/main.py
```
