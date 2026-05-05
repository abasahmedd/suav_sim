import numpy as np
from matplotlib import pyplot as plt

from DataDisplay.panel_components import Plot


class PositionActualEstimatedPlot(Plot):
    def __init__(
        self,
        fig: plt.Figure,
        pos: int = 111,
        is_3d: bool = False,
        waypoint_coords: np.ndarray | None = None,
        route_coords: np.ndarray | None = None,
        title: str = "Position in NED Frame",
    ) -> None:
        super().__init__(fig, pos, is_3d, nvars=6)
        self.waypoint_coords = waypoint_coords
        self.route_coords = route_coords

        self.ax.set_title(title, fontname="Times New Roman")
        self.ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        self.ax.set_xlabel("East (m)")
        self.ax.set_ylabel("North (m)")

        if is_3d:
            self.ax.set_zlabel("Height (m)")
            self.actual_line, = self.ax.plot([], [], [], color="k", linewidth=1.8, label="Actual")
            self.estimated_line, = self.ax.plot([], [], [], color="r", linestyle="--", linewidth=1.4, label="EKF")
            self.waypoint_points = None
        else:
            self.actual_line, = self.ax.plot([], [], color="k", linewidth=1.8, label="Actual")
            self.estimated_line, = self.ax.plot([], [], color="r", linestyle="--", linewidth=1.4, label="EKF")
            if route_coords is not None and route_coords.size > 0:
                self.route_line, = self.ax.plot(
                    route_coords[:, 1],
                    route_coords[:, 0],
                    color="b",
                    linestyle=":",
                    linewidth=1.2,
                    label="Reference route",
                    zorder=2,
                )
            else:
                self.route_line = None
            if waypoint_coords is not None and waypoint_coords.size > 0:
                self.waypoint_points = self.ax.scatter(
                    waypoint_coords[:, 1],
                    waypoint_coords[:, 0],
                    color="b",
                    s=30,
                    marker="o",
                    label="Waypoints",
                    zorder=3,
                )
                for index, waypoint in enumerate(waypoint_coords, start=1):
                    self.ax.text(
                        waypoint[1] + 8.0,
                        waypoint[0] + 8.0,
                        f"{int(index * 10)}",
                        fontsize=8,
                        color="b",
                        fontname="Times New Roman",
                    )
            else:
                self.waypoint_points = None

        if is_3d:
            self.route_line = None
        self.ax.legend(loc="best")
        self.ax.set_aspect("auto")
        self.logger.labels = ["pn", "pe", "pd", "pn_est", "pe_est", "pd_est"]
        render_objects = [self.actual_line, self.estimated_line]
        if self.route_line is not None:
            render_objects.append(self.route_line)
        if self.waypoint_points is not None:
            render_objects.append(self.waypoint_points)
        self.setup_blit(render_objects)

    def update_plot(self, render: bool = True) -> None:
        pos = self.logger.as_dict()

        pe_all = np.hstack((pos["pe"], pos["pe_est"]))
        pn_all = np.hstack((pos["pn"], pos["pn_est"]))
        h_all = np.hstack((-pos["pd"], -pos["pd_est"]))

        self.actual_line.set_data(pos["pe"], pos["pn"])
        self.estimated_line.set_data(pos["pe_est"], pos["pn_est"])
        if self.is_3d:
            self.actual_line.set_3d_properties(-pos["pd"])
            self.estimated_line.set_3d_properties(-pos["pd_est"])

        pe_min, pe_max = np.min(pe_all), np.max(pe_all)
        pn_min, pn_max = np.min(pn_all), np.max(pn_all)
        pe_span = max(pe_max - pe_min, 20.0)
        pn_span = max(pn_max - pn_min, 20.0)
        self.ax.set_xlim(pe_min - 0.1 * pe_span, pe_max + 0.1 * pe_span)
        self.ax.set_ylim(pn_min - 0.1 * pn_span, pn_max + 0.1 * pn_span)
        if self.is_3d:
            h_min, h_max = np.min(h_all), np.max(h_all)
            h_span = max(h_max - h_min, 10.0)
            self.ax.set_zlim(h_min - 0.1 * h_span, h_max + 0.1 * h_span)

        if render:
            self.render()
