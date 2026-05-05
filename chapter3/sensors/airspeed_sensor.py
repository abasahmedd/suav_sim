from dataclasses import dataclass

import numpy as np

from chapter1.aircraft_states import AircraftStates
from DataDisplay.isa import isa_density
from chapter3.sensors.sensor import Sensor, SensorParams
from chapter3.sensors.signal_simulation import saturate


@dataclass
class AirspeedParams(SensorParams):
    """
    Differential pressure sensor (airspeed) configuration handler, children of `SensorParams`

    Attributes
    ----------
    min_value : float, optional
        The minimum measurable differential pressure in kPa, by default 0.0
    max_value : float, optional
        The maximum measurable differential pressure in kPa, by default 4.0
    bias : float, optional
        The constant bias in kPa, by default 0.020
    deviation : float, optional
        The standard deviation in kPa, by default 0.002
    """

    sample_rate: float = 10.0  # Hz
    reading_delay: float = 0.0  # seconds
    min_value: float = 0.0
    max_value: float = 4.0
    bias: float = 0.020
    deviation: float = 0.002


class AirspeedSensor(Sensor):

    def __init__(
        self, params: AirspeedParams, state: AircraftStates, name: str = "asp"
    ) -> None:
        super().__init__(
            params,
            state,
            name=name,
            reading_names=["diff_pressure"],
            reading_units=["kPa"],
        )

        self.params = params

        self.ideal_value = np.zeros(1)
        self.noisy_value = np.zeros(1)
        self.prev_noisy_value = np.zeros(1)

    def get_ideal_value(self, t: float) -> np.ndarray:
        h = self.state.altitude
        rho = isa_density(h)
        Va = self.state.airspeed
        p = 0.5 * rho * Va**2
        return np.array([p]) * 1e-3  # Convert to kPa

    def get_noisy_value(self, t: float) -> np.ndarray:
        if self.ideal_value is None:
            raise ValueError("Ideal value has not been computed yet.")
        reading = self.ideal_value + np.random.normal(
            loc=self.params.bias, scale=self.params.deviation, size=1
        )
        reading = saturate(
            reading, min_val=self.params.min_value, max_val=self.params.max_value
        )
        return reading
