"""Main losses module."""
from abc import ABC, abstractmethod
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid, simpson
from piglot.losses.weights import UniformWeights


class Filter(ABC):
    """Abstract filter class."""

    @abstractmethod
    def extract(self, x, reference, prediction):
        """Extract filtered response from both reference and prediction fields.

        Parameters
        ----------
        x : array
            Time coordinate
        reference : array
            Reference response
        prediction : array
            Predicted response
        """


class Range(Filter):
    """Extract a range of values from responses."""

    def __init__(self, lbound=None, ubound=None):
        """Constructor for Range class.

        Parameters
        ----------
        lbound : float, optional
            Lower bound of the region to extract, by default None.
        ubound : float, optional
            Upper bound of the region to extract, by default None.
        """
        self.lbound = -np.inf if lbound is None else lbound
        self.ubound = np.inf if ubound is None else ubound

    def extract(self, x, reference, prediction):
        """Extract filtered response from both reference and prediction fields.

        Parameters
        ----------
        x : array
            Time coordinate
        reference : array
            Reference response
        prediction : array
            Predicted response
        """
        ifilt = np.logical_and(x >= self.lbound, x <= self.ubound)
        return x[ifilt], reference[ifilt], prediction[ifilt]


class Maximum(Filter):
    """Extract the maximum value of each field."""

    def extract(self, x, reference, prediction):
        """Extract filtered response from both reference and prediction fields.

        Parameters
        ----------
        x : array
            Time coordinate
        reference : array
            Reference response
        prediction : array
            Predicted response
        """
        ifilt = np.argmax(reference)
        ifilt2 = np.argmax(prediction)
        return x[ifilt], reference[ifilt], prediction[ifilt2]


class Minimum(Filter):
    """Extract the minimum value of each field."""

    def extract(self, x, reference, prediction):
        """Extract filtered response from both reference and prediction fields.

        Parameters
        ----------
        x : array
            Time coordinate
        reference : array
            Reference response
        prediction : array
            Predicted response
        """
        ifilt = np.argmin(reference)
        ifilt2 = np.argmin(prediction)
        return x[ifilt], reference[ifilt], prediction[ifilt2]



class Modifier(ABC):
    """Base modifier class to modify the reference and prediction responses."""

    @abstractmethod
    def modify(self, x, reference, prediction):
        """Modify both reference and prediction fields.

        Parameters
        ----------
        x : array
            Time coordinate
        reference : array
            Reference response
        prediction : array
            Predicted response
        """


class Weightings(Modifier):
    """Apply a weighting function to both reference and prediction reponses."""

    def __init__(self, weight=UniformWeights()):
        """Constructor for Weightings class.

        Parameters
        ----------
        weight : Weight, optional
            Weighting function to use, by default UniformWeights().
        """
        self.weight = weight

    def modify(self, x, reference, prediction):
        """Modify both reference and prediction fields.

        Parameters
        ----------
        x : array
            Time coordinate
        reference : array
            Reference response
        prediction : array
            Predicted response
        """
        weights = self.weight.get_weights(x)
        return x, weights * reference, weights * prediction


class Slope(Modifier):
    """Compute the slope of the reference and prediction fields."""

    def modify(self, x, reference, prediction):
        """Modify both reference and prediction fields.

        Parameters
        ----------
        x : array
            Time coordinate
        reference : array
            Reference response
        prediction : array
            Predicted response
        """
        delta_x = x[1:] - x[:-1]
        new_x = (x[1:] + x[:-1]) / 2
        slope_ref = (reference[1:] - reference[:-1]) / delta_x
        slope_pred = (prediction[1:] - prediction[:-1]) / delta_x
        return new_x, slope_ref, slope_pred


class Loss(ABC):
    """Abstract generic loss function."""

    def __init__(self, modifiers=[], filters=[], interp_kind='linear'):
        """Constructor for Loss class.

        Parameters
        ----------
        modifiers : list[Modifier], optional
            List of loss modifiers, by default []
        filters : list[Filter], optional
            List of loss filters, by default []
        interp_kind : str, optional
            Interpolation to use for slope computation, by default 'linear'.
            Refer to `scipy.interpolate.interp1d` documentation for details.
        """
        self.modifiers = modifiers
        self.filters = filters
        self.interp_kind = interp_kind

    @abstractmethod
    def _compute(self, x, reference, prediction):
        """Main method for loss computation.

        Parameters
        ----------
        x : array
            Time coordinate
        reference : array
            Reference response
        prediction : array
            Predicted response
        """

    @staticmethod
    def _reduction_func(reduction):
        """Returns the reduction function to use, depending on the flag.

        Parameters
        ----------
        reduction : str
            Reduction type to use. Possible values are:
            - `sum`: sum of the error function
            - `trapezoid`: integration of the error function with a trapezoid rule
            - `simpson`: integration of the error function with a simpson rule

        Returns
        -------
        callable
            Reduction function to use.

        Raises
        ------
        Exception
            If the reduction type is unknown.
        """
        if reduction == "sum":
            return lambda x, y: np.sum(y)
        elif reduction == 'trapezoid':
            return lambda x, y: trapezoid(y, x=x)
        elif reduction == 'simpson':
            return lambda x, y: simpson(y, x=x)
        else:
            raise Exception("Unknown reduction {0}!".format(reduction))

    def _apply_filters_modifiers(self, x, ref, pred):
        """Internal function to apply filters and modifiers to the response.

        Parameters
        ----------
        x : array
            Time coordinate
        ref : array
            Reference response
        pred : array
            Predicted response

        Returns
        -------
        x : array
            Time coordinate
        ref : array
            Filtered and modified reference response
        pred : array
            Filtered and modified predicted response
        """
        # Apply filters
        for filter in self.filters:
            x, ref, pred = filter.extract(x, ref, pred)
        # Apply modifiers
        for modifier in self.modifiers:
            x, ref, pred = modifier.modify(x, ref, pred)
        return x, ref, pred

    def __call__(self, x_ref, x_pred, y_ref, y_pred):
        """Loss computation function.

        Parameters
        ----------
        x_ref : array
            Reference time coordinates
        x_pred : array
            Prediction time coordinates
        y_ref : array
            Reference response
        y_pred : array
            Prediction response

        Returns
        -------
        float
            Loss value
        """
        # Interpolate prediction grid on the reference grid
        x = x_ref
        ref = y_ref
        func = interp1d(x_pred, y_pred, kind=self.interp_kind, bounds_error=False,
                        fill_value='extrapolate')
        pred = func(x_ref)
        # Filters and modifiers
        x, ref, pred = self._apply_filters_modifiers(x, ref, pred)
        # Compute the derived loss
        loss = self._compute(x, ref, pred)
        # Penalty if the predicted response is shorter than the reference one
        len_ref = np.max(x) - np.min(x)
        len_pred = np.max(x_pred) - np.min(x_pred)
        if len_pred < len_ref:
            factor = float(len_ref - len_pred) / len_ref
            loss += factor * self.max_value(x_ref, y_ref)
        return loss

    def max_value(self, x_ref, y_ref):
        """Returns the possible max value for this loss.

        Parameters
        ----------
        x_ref : array
            Reference time coordinates
        y_ref : array
            Reference response

        Returns
        -------
        float
            Maximum loss value.
        """
        return self._compute(x_ref, y_ref, 0.0 * y_ref)


class BaseLoss(Loss):
    """Base class for generic losses based on the difference of responses."""

    def __init__(self, err_squared=False, reduction='sum', normalise='ref', n_points=True,
                 reshaper='none', *args, **kwargs):
        """Constructor for BaseLoss

        Parameters
        ----------
        err_squared : bool, optional
            Whether to use the squares of the error or the absolute value, by default False.
        reduction : str, optional
            Reduction type to use, by default 'sum'.
            Refer to Loss._reduction_func().
        normalise : str, optional
            Whether to normalise loss value, by default 'ref'.
            Possible values are:
            - `none`: No normalisation
            - `ref`: Normalise by reference response
            - `pred`: Normalise by predicted response
        n_points : bool, optional
            Whether to divide by the number of points in the reference response, by default
            True
        reshaper : str, optional
            Reshape function to apply to the loss, by default 'none'. Possible values:
            - `none`: No reshaping
            - `sqrt`: Apply the square root to the loss value
            - `log`: Apply the natural logarithm to the loss value
        """
        super().__init__(*args, **kwargs)
        assert normalise in ['none', 'ref', 'pred']
        assert reshaper in ['none', 'sqrt', 'log']
        self.err_squared = err_squared
        self.reduction = reduction
        self.normalise = normalise
        self.n_points = n_points
        self.reshaper = reshaper

    def _compute(self, x, reference, prediction):
        """Main method for loss computation.

        Parameters
        ----------
        x : array
            Time coordinate
        reference : array
            Reference response
        prediction : array
            Predicted response
        """
        err_func = lambda y: np.square(y) if self.err_squared else np.abs(y)
        reduction_func = self._reduction_func(self.reduction)
        if self.normalise == 'none':
            norm_func = lambda y_ref, y_pred: 1
        elif self.normalise == 'ref':
            norm_func = lambda y_ref, y_pred: reduction_func(x, err_func(y_ref))
        elif self.normalise == 'pred':
            norm_func = lambda y_ref, y_pred: reduction_func(x, err_func(y_pred))
        n_points = 1
        if self.n_points:
            try:
                n_points = len(x)
            except:
                n_points = 1
        if self.reshaper == 'none':
            shape_func = lambda y: y
        elif self.reshaper == 'sqrt':
            shape_func = lambda y: np.sqrt(y)
        elif self.reshaper == 'log':
            shape_func = lambda y: np.log(y)

        return shape_func(reduction_func(x, err_func(reference - prediction)) 
                          / norm_func(reference, prediction) / n_points)


class MSE(BaseLoss):
    """Mean squared error Loss."""

    def __init__(self, reduction='sum', *args, **kwargs):
        """Constructor for this Loss.

        Parameters
        ----------
        reduction : str, optional
            Reduction function to apply, by default 'sum'. Refer to BaseLoss.
        """
        super().__init__(err_squared=True, reduction=reduction, normalise='none',
                         n_points=True, reshaper='none', *args, **kwargs)


class RNMSE(BaseLoss):
    """Root normalised mean squared error Loss."""

    def __init__(self, reduction='sum', *args, **kwargs):
        """Constructor for this Loss.

        Parameters
        ----------
        reduction : str, optional
            Reduction function to apply, by default 'sum'. Refer to BaseLoss.
        """
        super().__init__(err_squared=True, reduction=reduction, normalise='ref',
                         n_points=True, reshaper='sqrt', *args, **kwargs)


class NMSE(BaseLoss):
    """Normalised mean squared error Loss."""

    def __init__(self, reduction='sum', *args, **kwargs):
        """Constructor for this Loss.

        Parameters
        ----------
        reduction : str, optional
            Reduction function to apply, by default 'sum'. Refer to BaseLoss.
        """
        super().__init__(err_squared=True, reduction=reduction, normalise='ref',
                         n_points=True, reshaper='none', *args, **kwargs)


class LogNMSE(NMSE):
    """Logarithmic normalised mean squared error Loss."""

    def __init__(self, reduction='sum', *args, **kwargs):
        """Constructor for this Loss.

        Parameters
        ----------
        reduction : str, optional
            Reduction function to apply, by default 'sum'. Refer to BaseLoss.
        """
        super().__init__(reduction, *args, **kwargs)
    
    def _compute(self, x, reference, prediction):
        """Main method for loss computation.

        Parameters
        ----------
        x : array
            Time coordinate
        reference : array
            Reference response
        prediction : array
            Predicted response
        """
        return np.log(super()._compute(x, reference, prediction))

class RMSE(BaseLoss):
    """Root mean squared error Loss."""

    def __init__(self, reduction='sum', *args, **kwargs):
        """Constructor for this Loss.

        Parameters
        ----------
        reduction : str, optional
            Reduction function to apply, by default 'sum'. Refer to BaseLoss.
        """
        super().__init__(err_squared=True, reduction=reduction, normalise='none',
                         n_points=True, reshaper='sqrt', *args, **kwargs)


class MAE(BaseLoss):
    """Mean absolute error Loss."""

    def __init__(self, reduction='sum', *args, **kwargs):
        """Constructor for this Loss.

        Parameters
        ----------
        reduction : str, optional
            Reduction function to apply, by default 'sum'. Refer to BaseLoss.
        """
        super().__init__(err_squared=False, reduction=reduction, normalise='none',
                         n_points=True, reshaper='none', *args, **kwargs)


class RNMAE(BaseLoss):
    """Root normalised mean absolute error Loss."""

    def __init__(self, reduction='sum', *args, **kwargs):
        """Constructor for this Loss.

        Parameters
        ----------
        reduction : str, optional
            Reduction function to apply, by default 'sum'. Refer to BaseLoss.
        """
        super().__init__(err_squared=False, reduction=reduction, normalise='ref',
                         n_points=True, reshaper='sqrt', *args, **kwargs)


class NMAE(BaseLoss):
    """Normalised mean absolute error Loss."""

    def __init__(self, reduction='sum', *args, **kwargs):
        """Constructor for this Loss.

        Parameters
        ----------
        reduction : str, optional
            Reduction function to apply, by default 'sum'. Refer to BaseLoss.
        """
        super().__init__(err_squared=False, reduction=reduction, normalise='ref',
                         n_points=True, reshaper='none', *args, **kwargs)


class RMAE(BaseLoss):
    """Root mean absolute error Loss."""
    def __init__(self, reduction='sum', *args, **kwargs):
        """Constructor for this Loss.

        Parameters
        ----------
        reduction : str, optional
            Reduction function to apply, by default 'sum'. Refer to BaseLoss.
        """
        super().__init__(err_squared=False, reduction=reduction, normalise='none',
                         n_points=True, reshaper='sqrt', *args, **kwargs)


class MeanNRMSE(Loss):
    """Mean normalised root mean squared error."""

    def __init__(self, reduction="sum", *args, **kwargs):
        """Constructor for this Loss.

        Parameters
        ----------
        reduction : str, optional
            Reduction function to apply, by default 'sum'. Refer to BaseLoss.
        """
        super().__init__(*args, **kwargs)
        self.reduction = reduction

    def _compute(self, x, reference, prediction):
        """Main method for loss computation.

        Parameters
        ----------
        x : array
            Time coordinate
        reference : array
            Reference response
        prediction : array
            Predicted response
        """
        reduction_func = self._reduction_func(self.reduction)
        try:
            n_points = len(x)
        except:
            n_points = 1
        mean = abs(np.mean(reference))
        return np.sqrt(reduction_func(x, np.square(reference - prediction)) / n_points) / mean


class MixedLoss(Loss):
    """Base class for mixing losses."""

    def __init__(self, *args, **kwargs):
        """Constructor for MixedLoss."""
        super().__init__(*args, **kwargs)
        self.losses = []

    def add_loss(self, loss, ratio):
        """Add a loss to the MixedLoss.

        Parameters
        ----------
        loss : Loss
            Loss to add.
        ratio : float
            Weight for this loss.
        """
        self.losses.append((ratio, loss))

    def _compute(self, x, reference, prediction):
        """Main method for loss computation.

        Parameters
        ----------
        x : array
            Time coordinate
        reference : array
            Reference response
        prediction : array
            Predicted response
        """
        value = 0
        for loss in self.losses:
            x_new, ref_new, pred_new = loss[1]._apply_filters_modifiers(x, reference, prediction)
            value += loss[0] * loss[1]._compute(x_new, ref_new, pred_new)
        return value
