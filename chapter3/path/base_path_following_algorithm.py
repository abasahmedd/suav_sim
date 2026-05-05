
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from chapter2.autopilot_design_parameters import AutopilotDesignParameters


@dataclass
class BasePathParams(ABC):
    """Base class for path parameters."""

    pass


class BasePathFollower(ABC):
    """Base class for path following guidance."""

    def __init__(self, config: AutopilotDesignParameters) -> None:
        """
        Initialize the PathGuidance with autopilot configuration.

        Parameters
        ----------
        config : AutopilotDesignParameters
            Configuration parameters for the autopilot.
        """
        self.config = config

    @abstractmethod
    def set_path(self, *args, **kwargs):
        """
        Set the path parameters. To be implemented by subclasses.
        """
        pass

    @abstractmethod
    def lateral_guidance(self, pos_ned: np.ndarray, course: float = 0.0) -> float:
        """
        Calculate the lateral guidance.

        Parameters
        ----------
        pos_ned : np.ndarray
            A 3-element array representing the position in NED (North-East-Down) frame.
        course : float, optional
            Current course angle of the aircraft (default is 0.0).

        Returns
        -------
        float
            The reference course angle.
        """
        pass

    @abstractmethod
    def longitudinal_guidance(self, pos_ned: np.ndarray) -> float:
        """
        Calculate the longitudinal guidance.

        Parameters
        ----------
        pos_ned : np.ndarray
            A 3-element array representing the position in NED (North-East-Down) frame.

        Returns
        -------
        float
            The reference altitude.
        """
        pass

    def guidance(self, pos_ned: np.ndarray, course: float = 0.0) -> tuple[float, float]:
        """
        Provide both lateral and longitudinal guidance.

        Parameters
        ----------
        pos_ned : np.ndarray
            A 3-element array representing the position in NED (North-East-Down) frame.
        course : float, optional
            Current course angle of the aircraft (default is 0.0).

        Returns
        -------
        tuple[float, float]
            The reference course angle and altitude for path following as `(course_ref, altitude_ref)`.
        """
        course_ref = self.lateral_guidance(pos_ned, course)
        altitude_ref = self.longitudinal_guidance(pos_ned)
        return course_ref, altitude_ref

