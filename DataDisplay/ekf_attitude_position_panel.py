from typing_extensions import override

import numpy as np

from chapter1.aircraft_states import AircraftStates
from chapter2.autopilot_status_monitoring import AutopilotStatusMonitoring
from DataDisplay.attitude_view import AttitudeView
from DataDisplay.panel_base import Panel
from DataDisplay.position_actual_estimated_plot import PositionActualEstimatedPlot
from DataDisplay.time_series_plot import TimeSeriesPlot


class EkfAttitudePositionPanel(Panel):
    def __init__(
        self,
        figsize=(15, 8),
        use_blit: bool = False,
        pos_3d: bool = True,
        waypoint_coords: np.ndarray | None = None,
        route_coords: np.ndarray | None = None,
    ) -> None:
        super().__init__(figsize, use_blit)
        self.fig.set_facecolor("white")

        self.attitude_view = AttitudeView(self.fig, pos=231)
        self.top_view_plot = PositionActualEstimatedPlot(
            self.fig,
            pos=232,
            is_3d=False,
            waypoint_coords=waypoint_coords,
            route_coords=route_coords,
            title="Top View with Waypoints and Fillet Tracking",
        )
        self.position_plot = PositionActualEstimatedPlot(
            self.fig,
            pos=233,
            is_3d=pos_3d,
            title="3D Position in NED Frame",
        )
        self.roll_plot = TimeSeriesPlot(
            self.fig,
            pos=234,
            ylabel="Roll (deg)",
            xlabel="Time (s)",
            nvars=3,
            labels=["Actual - black", "Reference - blue", "EKF - red dashed"],
        )
        self.pitch_plot = TimeSeriesPlot(
            self.fig,
            pos=235,
            ylabel="Pitch (deg)",
            xlabel="Time (s)",
            nvars=3,
            labels=["Actual - black", "Reference - blue", "EKF - red dashed"],
        )
        self.empty_ax = self.fig.add_subplot(236)
        self.empty_ax.axis("off")

        self.add_components(
            [
                self.attitude_view,
                self.top_view_plot,
                self.position_plot,
                self.roll_plot,
                self.pitch_plot,
            ]
        )
        self._style_plots()
        self.fig.tight_layout(pad=2)
        self._expand_3d_plot()

    def _expand_3d_plot(self) -> None:
        pos3d = self.position_plot.ax.get_position()
        pos_empty = self.empty_ax.get_position()
        x0 = min(pos3d.x0, pos_empty.x0)
        y0 = min(pos3d.y0, pos_empty.y0)
        x1 = max(pos3d.x1, pos_empty.x1)
        y1 = max(pos3d.y1, pos_empty.y1)
        self.position_plot.ax.set_position([x0, y0, x1 - x0, y1 - y0])
        self.empty_ax.set_visible(False)

    def _style_plots(self) -> None:
        for axis in [self.roll_plot.ax, self.pitch_plot.ax]:
            axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
            axis.set_facecolor("white")
            axis.title.set_fontname("Times New Roman")
            axis.xaxis.label.set_fontname("Times New Roman")
            axis.yaxis.label.set_fontname("Times New Roman")
            for label in axis.get_xticklabels() + axis.get_yticklabels():
                label.set_fontname("Times New Roman")

        for axis in [self.top_view_plot.ax, self.position_plot.ax]:
            axis.set_facecolor("white")
            axis.title.set_fontname("Times New Roman")
            axis.xaxis.label.set_fontname("Times New Roman")
            axis.yaxis.label.set_fontname("Times New Roman")

        self.roll_plot.lines[0].set_color("k")
        self.roll_plot.lines[0].set_linewidth(1.5)
        self.roll_plot.lines[1].set_color("b")
        self.roll_plot.lines[1].set_linewidth(1.2)
        self.roll_plot.lines[1].set_linestyle("-")
        self.roll_plot.lines[1].set_label("Reference")
        self.roll_plot.lines[2].set_color("r")
        self.roll_plot.lines[2].set_linewidth(1.2)
        self.roll_plot.lines[2].set_linestyle("--")
        self.roll_plot.lines[2].set_label("EKF")
        self.roll_plot.lines[0].set_label("Actual")

        self.pitch_plot.lines[0].set_color("k")
        self.pitch_plot.lines[0].set_linewidth(1.5)
        self.pitch_plot.lines[1].set_color("b")
        self.pitch_plot.lines[1].set_linewidth(1.2)
        self.pitch_plot.lines[1].set_linestyle("-")
        self.pitch_plot.lines[1].set_label("Reference")
        self.pitch_plot.lines[2].set_color("r")
        self.pitch_plot.lines[2].set_linewidth(1.2)
        self.pitch_plot.lines[2].set_linestyle("--")
        self.pitch_plot.lines[2].set_label("EKF")
        self.pitch_plot.lines[0].set_label("Actual")

        self.roll_plot.ax.legend(
            self.roll_plot.lines,
            [line.get_label() for line in self.roll_plot.lines],
            loc="upper right",
            fontsize=8,
            frameon=True,
        )
        self.pitch_plot.ax.legend(
            self.pitch_plot.lines,
            [line.get_label() for line in self.pitch_plot.lines],
            loc="upper right",
            fontsize=8,
            frameon=True,
        )

    def add_data(
        self,
        time: float,
        state: AircraftStates,
        ap_status: AutopilotStatusMonitoring,
        estimated_position_ned: np.ndarray,
        estimated_roll: float,
        estimated_pitch: float,
    ) -> None:
        combined_position = np.hstack((state.ned_position, estimated_position_ned))
        self.top_view_plot.add_data(combined_position, time)
        self.position_plot.add_data(
            combined_position,
            time,
        )
        self.roll_plot.add_data(
            np.rad2deg([state.roll, ap_status.target_roll, estimated_roll]),
            time,
        )
        self.pitch_plot.add_data(
            np.rad2deg([state.pitch, ap_status.target_pitch, estimated_pitch]),
            time,
        )

    def update_plots(self) -> None:
        self.top_view_plot.update_plot()
        self.position_plot.update_plot()
        self.roll_plot.update_plot()
        self.pitch_plot.update_plot()

    def update_views(self, state: AircraftStates) -> None:
        self.attitude_view.update_view(state.attitude_angles)

    @override
    def update(self, state: AircraftStates, pause: float = 0.01) -> None:
        return super().update(state=state, pause=pause)
