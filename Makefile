ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
$(eval $(ARGS):;@:)

init:
	pipenv install --dev
test:
	pipenv run python -m pytest $(ARGS)
example:
	pipenv run python examples/$(ARGS)
clean:
	pipenv run python setup.py clean
fmt:
	black .
