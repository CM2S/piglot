# Contributing to `piglot`

Thank you for considering contributing to `piglot`!
We appreciate your interest in helping us improve our project.

## How to Contribute

To contribute to `piglot`, please follow these steps:

1. Fork the repository and create a new branch for your contribution.
2. Make your changes or additions to the codebase.
3. Write tests to ensure the correctness of your changes.
4. Run the existing tests to make sure they pass.
5. If relevant, build a set of appropriate examples for the new features and add them to the documentation.
6. Commit your changes and push them to your forked repository.
7. Submit a pull request to the main repository.

### Tests

When contributing with new features, also write tests for them in the `test` directory.
We use `pytest` for this task in our CI framework.
While we currently don't have strict test coverage requirements, new features should be extensively tested.

### Examples

We strongly encourage contributors to also build minimal working examples (MWEs) for new features or solvers.
These should be placed inside the `examples` directory, along with the required additional files.
A description of the example should also be added to the `docs/source/examples` directory, showing the usage and results of your example.
Don't forget to add a reference to the indices in `docs/source/examples/*.rst`.

### Input file templates

If your new features require modifications to the `.yaml` configuration specification, these changes should be included in the input file templates.
These are hosted in the `examples/templates` directory, which are then loaded by the documentation in `docs/source/templates`.
If you add new solvers or optimisers, please update both locations with your new template.

## Code Style

Please adhere to the following code style guidelines when contributing to `piglot`:

- Use spaces for indentation. We are using 4 spaces.
- Follow the naming conventions used in the existing codebase.
- Write clear and concise comments to explain your code.
- Write docstrings for every module, method and class. We use the `numpy` docstring style.

We use `flake8` for linting the code.
Please refer to our configuration (`.flake8` in the repo root) for additional details.
Compliance with this is enforced in our CI procedure.

## Reporting Issues

If you encounter any issues or have suggestions for improvement, please open an issue on the [issue tracker](https://github.com/CM2S/piglot/issues).
Provide as much detail as possible to help us understand and address the problem.

## Papers using `piglot`

If you use `piglot` in your work, we encourage you to add your contribution to the [list of papers](docs/source/papers.md) using `piglot`.
Feel free to open a PR for adding an entry to that list!

## License

By contributing to `piglot`, you agree that your contributions will be licensed under the [MIT License](https://opensource.org/licenses/MIT).

We look forward to your contributions! Thank you for your support.
