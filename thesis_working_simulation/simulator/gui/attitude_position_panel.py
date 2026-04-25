"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""
from typing import Any
from typing_extensions import override

from simulator.aircraft import AircraftState
from simulator.gui.attitude_view import AttitudeView
from simulator.gui.position_plot import PositionPlot
from simulator.gui.panel_base import Panel
from simulator.gui.dashboard_view import DashboardView
from simulator.utils.simulation_data import SimulationData


class AttitudePositionPanel(Panel):
    """
    Panel that displays attitude and position data of the aircraft.

    Attributes
    ----------
    attitude_view : AttitudeView
        The attitude view component.
    position_plot : PositionPlot
        The position plot component.
    """

    def __init__(self, figsize=(12, 8), use_blit: bool = False, pos_3d: bool = True) -> None:
        """
        Initialize the AttitudePositionPanel with attitude, position, and dashboard views.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size of the panel, by default (12, 8).
        use_blit : bool, optional
            Whether to use blitting for rendering, by default False.
        pos_3d : bool, optional
            Whether to display the position in 3D, by default True.
        """
        super().__init__(figsize, use_blit)

        gs = self.fig.add_gridspec(2, 2, height_ratios=[3, 1])
        
        self.attitude_view = AttitudeView(self.fig, pos=gs[0, 0])
        self.position_plot = PositionPlot(self.fig, pos=gs[0, 1], is_3d=pos_3d)
        self.dashboard_view = DashboardView(self.fig, pos=gs[1, :])

        self.add_components([self.attitude_view, self.position_plot, self.dashboard_view])
        self.fig.tight_layout(pad=2)

    def add_data(self, data: SimulationData) -> None:
        """
        Add data to the attitude and position views.

        Parameters
        ----------
        data : SimulationData
            The current simulation data.
        """
        if data is not None:
            self.position_plot.add_data(data.uav_state.ned_position)

    def add_waypoints(self, waypoints: list) -> None:
        """
        Pass waypoints to the position plot.

        Parameters
        ----------
        waypoints : list
            List of Waypoint objects.
        """
        self.position_plot.add_waypoints(waypoints)

    def update_plots(self) -> None:
        """
        Update the position plot.
        """
        self.position_plot.update_plot()

    def update_views(self, data: SimulationData) -> None:
        """
        Update the attitude and dashboard views.

        Parameters
        ----------
        data : SimulationData
            The current simulation data.
        """
        if data is not None:
            self.attitude_view.update_view(data.uav_state.attitude_angles)
            self.dashboard_view.update_view(data)

    @override
    def update(self, data: SimulationData, pause: float = 0.01) -> None:
        """
        Update the views and plots.

        Parameters
        ----------
        data : SimulationData
            The current simulation data.
        pause : float, optional
            Time in seconds to pause the plot update, by default 0.01
        """
        return super().update(data, pause=pause)
