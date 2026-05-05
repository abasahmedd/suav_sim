from typing import Any
from typing_extensions import override

from chapter1.aircraft_states import AircraftStates
from chapter1.control_deflections import ControlDeflections
from DataDisplay.attitude_view import AttitudeView
from DataDisplay.horizontal_bar_view import HorizontalBarView
from DataDisplay.panel_base import Panel
from DataDisplay.position_plot import PositionPlot
from DataDisplay.time_series_plot import TimeSeriesPlot


class MainStatusPanel(Panel):
    """
    Panel that displays the main status of the aircraft including attitude,
    position, control deltas, altitude, and airspeed.

    Attributes
    ----------
    attitude_view : AttitudeView
        The attitude view component.
    position_plot : PositionPlot
        The position plot component.
    delta_a_view : HorizontalBarView
        The horizontal bar view for aileron deflection.
    delta_e_view : HorizontalBarView
        The horizontal bar view for elevator deflection.
    delta_r_view : HorizontalBarView
        The horizontal bar view for rudder deflection.
    delta_t_view : HorizontalBarView
        The horizontal bar view for throttle position.
    altitude_plot : TimeSeriesPlot
        The time series plot for altitude.
    airspeed_plot : TimeSeriesPlot
        The time series plot for airspeed.
    """

    def __init__(self, figsize=(10, 6), use_blit: bool = False, **kwargs) -> None:
        """
        Initialize the MainStatusPanel with various views and plots.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size of the panel, by default (10, 6).
        use_blit : bool, optional
            Whether to use blitting for rendering, by default False.
        """
        super().__init__(figsize, use_blit)
        self.attitude_view = AttitudeView(self.fig, pos=221)
        self.position_plot = PositionPlot(self.fig, pos=222, is_3d=False)
        self.delta_a_view = HorizontalBarView(
            self.fig, (8, 2, 9), "Aileron", (-1.0, +1.0)
        )
        self.delta_e_view = HorizontalBarView(
            self.fig, (8, 2, 11), "Elevator", (-1.0, +1.0)
        )
        self.delta_r_view = HorizontalBarView(
            self.fig, (8, 2, 13), "Rudder", (-1.0, +1.0)
        )
        self.delta_t_view = HorizontalBarView(
            self.fig, (8, 2, 15), "Throttle", (0.0, 1.0)
        )
        self.altitude_plot = TimeSeriesPlot(
            self.fig, pos=426, ylabel="Altitude (m)", title="Altitude & AirspeedSensor Log"
        )
        self.airspeed_plot = TimeSeriesPlot(
            self.fig, pos=428, ylabel="AirspeedSensor (m/s)", xlabel="Time (s)"
        )
        self.add_components(
            [
                self.attitude_view,
                self.position_plot,
                self.delta_a_view,
                self.delta_e_view,
                self.delta_r_view,
                self.delta_t_view,
                self.altitude_plot,
                self.airspeed_plot,
            ]
        )
        self.fig.subplots_adjust(wspace=0.4, hspace=0.4)

    def add_data(self, time: float, state: AircraftStates) -> None:
        """
        Add data to the main status views and plots.

        Parameters
        ----------
        time : float
            The current time.
        state : AircraftStates
            The current state of the aircraft.

        Raises
        ------
        ValueError
            If the required keyword argument 'time' or 'state' is not provided.
        """
        self.position_plot.add_data(state.ned_position)
        self.altitude_plot.add_data(state.altitude, time)
        self.airspeed_plot.add_data(state.airspeed, time)

    def update_plots(self) -> None:
        """
        Update the position, altitude, and airspeed plots.
        """
        self.position_plot.update_plot()
        self.altitude_plot.update_plot()
        self.airspeed_plot.update_plot()

    def update_views(self, state: AircraftStates, deltas: ControlDeflections) -> None:
        """
        Update the attitude and control deltas views.

        Parameters
        ----------
        state : AircraftStates
            The current state of the aircraft.
        deltas : ControlDeflections
            The current control surface deflections.

        Raises
        ------
        ValueError
            If the required keyword argument 'state' or 'deltas' is not provided.
        """
        self.attitude_view.update_view(state.attitude_angles)
        self.delta_a_view.update_view(deltas.delta_a)
        self.delta_e_view.update_view(deltas.delta_e)
        self.delta_r_view.update_view(deltas.delta_r)
        self.delta_t_view.update_view(deltas.delta_t)

    @override
    def update(self, state: AircraftStates, deltas: ControlDeflections, pause: float = 0.01) -> None:
        """
        Update the attitude and control deltas view;
        and the position, altitude, and airspeed plots.

        Parameters
        ----------
        state : AircraftStates
            The current state of the aircraft.
        deltas : ControlDeflections
            The current control surface deflections.
        pause : float, optional
            Time in seconds to pause the plot update, by default 0.01

        Raises
        ------
        ValueError
            If the required keyword argument 'state' or 'deltas' is not provided.
        """
        return super().update(state=state, deltas=deltas, pause=pause)
