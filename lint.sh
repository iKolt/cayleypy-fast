result=0

echo "Running black..."
black --check ./cayleypy_fast ./tests ./conftest.py
result+=$?

echo "Running pylint..."
pylint ./cayleypy_fast
result+=$?

echo "Running mypy..."
mypy ./cayleypy_fast
result+=$?

exit $result
