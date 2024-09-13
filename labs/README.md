# CS 5330 Labs and Activity Learnings

## How to run the app
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
