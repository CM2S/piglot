"""Plotting utilities for piglot."""
import os.path
import numpy as np
import matplotlib.pyplot as plt
from yaml import safe_load_all
from yaml.parser import ParserError
from piglot.links import LinksCase
from piglot.yaml_parser import parse_case


class Response:
    """Container for responses."""

    def __init__(self, field_data):
        """Constructor for response container.

        Parameters
        ----------
        field_data : array
            Two-dimensional array with the field data.
        """
        self.x = [x[0] for x in field_data]
        self.y = [x[1] for x in field_data]


class CaseData:
    """Container for case data."""

    def __init__(self, filename):
        """Constructor for case data container.

        Parameters
        ----------
        filename : str
            Path for the case output file.

        Raises
        ------
        Exception
            If the case file fails to parse.
        """
        with open(filename, 'r') as file:
            try:
                info, fields = safe_load_all(file)
            except ParserError:
                raise Exception("Failed to parse the case file!")
        self.filename = info["filename"]
        self.loss = float(info["loss"])
        self.parameters = info["parameters"]
        self.run_time = info["run_time"]
        self.start_time = info["start_time"]
        self.success = info["success"] == "true"
        self.fields = {name: Response(data) for name, data in fields.items()}


def plot_case_data(case: CaseData, title=None):
    """Plot a single case.

    Parameters
    ----------
    case : CaseData
        Case to plot
    title : str, optional
        Title of the figure, by default None

    Returns
    -------
    Figure, dict[Axes]
        Figure and dict with figure axes

    Raises
    ------
    Exception
        If there are no fields to load
    """
    n_fields = len(case.fields)
    if n_fields == 0:
        raise Exception("No fields to output have been found. Check if simulation completed.")
    n_cols = min(max(1, n_fields), 2)
    n_rows = int(np.ceil(n_fields / 2))
    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False)
    axes = [a for b in axes for a in b]
    axes_dict = {}
    for i, (name, data) in enumerate(case.fields.items()):
        axes[i].plot(data.x, data.y, label='Prediction', c='red')
        axes[i].set_title(name)
        axes[i].grid()
        axes_dict[name] = axes[i]
    if title:
        fig.suptitle(title)
    return fig, axes_dict


def plot_reference_data(case_data: CaseData, config, fig, axes_dict):
    """Append reference data to a plot.

    Parameters
    ----------
    case_data : CaseData
        Case to plot
    config : dict
        Configuration settings
    fig : Figure
        Figure to plot to
    axes_dict : dict[Axes]
        Dict with the axes to plot to

    Returns
    -------
    Figure, dict[Axes]
        Figure and dict with figure axes
    """
    case = parse_case(case_data.filename, config["cases"][case_data.filename])
    for field, reference in case.fields.items():
        axis = axes_dict[field.name()]
        axis.plot(reference[:, 0], reference[:, 1],
                  label='Reference', ls='dashed', c='black', marker='x')
        axis.legend()
    return fig, axes_dict



class CurrentPlot:
    """Container for dynamically-updating plots."""

    def __init__(self, case: LinksCase, config):
        """Constructor for dynamically-updating plots

        Parameters
        ----------
        case : LinksCase
            Case to plot
        config : dict
            Configuration settings
        """
        self.case = case
        n_fields = len(case.fields)
        n_cols = min(max(1, n_fields), 2)
        n_rows = int(np.ceil(n_fields / 2))
        self.fig, axes = plt.subplots(n_rows, n_cols, squeeze=False)
        self.axes = [a for b in axes for a in b]
        tmp_dir = config["tmp_dir"] if "tmp_dir" in config else os.path.join(config["output"], "tmp")
        self.path = os.path.join(tmp_dir, case.filename)
        # Make initial plot
        self.pred = {}
        for i, (field, reference) in enumerate(self.case.fields.items()):
            name = field.name()
            data = field.get(self.path)
            self.axes[i].plot(reference[:, 0], reference[:, 1],
                              label='Reference', ls='dashed', c='black', marker='x')
            self.pred[name], = self.axes[i].plot(data[:, 0], data[:, 1],
                                                 label='Prediction', c='red')
            self.axes[i].set_title(name)
            self.axes[i].grid()
            self.axes[i].legend()
        self.fig.suptitle(case.filename)
        plt.show()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(self):
        """Update the plot with the most recent prediction."""
        for i, field in enumerate(self.case.fields.keys()):
            name = field.name()
            try:
                data = field.get(self.path)
                self.pred[name].set_xdata(data[:, 0])
                self.pred[name].set_ydata(data[:, 1])
                self.axes[i].relim()
                self.axes[i].autoscale_view()
            except FileNotFoundError:
                pass
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
