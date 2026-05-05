import numpy as np

from .attitude_extended_kalman_filter import AttitudeExtendedKalmanFilter
from .state_estimation_filter import StateEstimationFilter, EstimatedState
from chapter3.sensors.integrated_sensor_system import SensorReadings


class KalmanEstimator(StateEstimationFilter):
    def __init__(self, dt: float) -> None:
        super().__init__(dt)
        self.attitde_ekf = AttitudeExtendedKalmanFilter(dt)
        
    def update(self, readings: SensorReadings) -> EstimatedState:

        return None
