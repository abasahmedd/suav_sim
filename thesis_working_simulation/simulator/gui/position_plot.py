"""
 Copyright (c) 2024 Pablo Ramirez Escudero
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
"""

import numpy as np
from matplotlib import pyplot as plt

from simulator.gui.panel_components import Plot
from simulator.math.rotation import ned2xyz


class PositionPlot(Plot):
    """
    A class to create and manage a plot for visualizing position data 
    in the North-East-Down (NED) coordinate frame.

    Attributes
    ----------
    line : matplotlib.lines.Line2D
        The line object representing the position in the plot.
    """

    def __init__(self, fig: plt.Figure, pos: int = 111, is_3d: bool = False) -> None:
        """
        Initializes the PositionPlot component.

        Parameters
        ----------
        fig : plt.Figure
            The matplotlib figure object to which this plot belongs.
        pos : int, optional
            The position of the subplot within the figure, by default 111.
        is_3d : bool, optional
            Whether the subplot is 3D, by default False.
        """
        super().__init__(fig, pos, is_3d, nvars=3)
                
        self.ax.set_title("Position in NED Frame")
        self.ax.grid()
        self.ax.set_xlabel("East (m)")
        self.ax.set_ylabel("North (m)")
        if is_3d:
            self.ax.set_zlabel("Height (m)")
            self.line, = self.ax.plot([], [], [], color='b', linewidth=2.0)
        else:
            self.line, = self.ax.plot([], [], color="b", linewidth=2.0)
        self.ax.set_aspect('auto')
        
        self.logger.labels = ["pn", "pe", "pd"]
            
        self.setup_blit([self.line])

    def add_waypoints(self, waypoints: list) -> None:
        """
        Plot the target waypoints on the position plot.

        Parameters
        ----------
        waypoints : list
            List of Waypoint objects.
        """
        if not waypoints:
            return
            
        pe_list = [wp.pe for wp in waypoints]
        pn_list = [wp.pn for wp in waypoints]
        
        if self.is_3d:
            pd_list = [wp.h for wp in waypoints] # h is positive up, same as -pd
            self.ax.plot(pe_list, pn_list, pd_list, 'r*--', linewidth=1.5, markersize=8, alpha=0.8, label='Waypoints')
            self.wp_min_pd = min(pd_list)
            self.wp_max_pd = max(pd_list)
        else:
            self.ax.plot(pe_list, pn_list, 'r*--', linewidth=1.5, markersize=8, alpha=0.8, label='Waypoints')
            
        self.wp_min_pe = min(pe_list)
        self.wp_max_pe = max(pe_list)
        self.wp_min_pn = min(pn_list)
        self.wp_max_pn = max(pn_list)
        self.ax.legend()

    def update_plot(self, render: bool = True) -> None:
        """
        Updates the plot with the latest position data from the logger.

        The method fetches the position data from the logger and updates
        the line object in the plot. It also adjusts the axis limits to fit
        the new data, including adjustments for 3D plots if necessary.
        """
        pos = self.logger.as_dict()
        
        self.line.set_data(pos["pe"], pos["pn"])
        if self.is_3d:
            self.line.set_3d_properties(-pos["pd"])
        
        # Autoscale axis limits
        min_pe = np.min(pos["pe"])
        max_pe = np.max(pos["pe"])
        min_pn = np.min(pos["pn"])
        max_pn = np.max(pos["pn"])
        
        if hasattr(self, 'wp_min_pe'):
            min_pe = min(min_pe, self.wp_min_pe)
            max_pe = max(max_pe, self.wp_max_pe)
            min_pn = min(min_pn, self.wp_min_pn)
            max_pn = max(max_pn, self.wp_max_pn)
            
        self.ax.set_xlim(min_pe - 100, max_pe + 100)
        self.ax.set_ylim(min_pn - 100, max_pn + 100)
        
        if self.is_3d:
            min_pd = np.min(-pos["pd"])
            max_pd = np.max(-pos["pd"])
            if hasattr(self, 'wp_min_pd'):
                min_pd = min(min_pd, self.wp_min_pd)
                max_pd = max(max_pd, self.wp_max_pd)
            self.ax.set_zlim(min_pd - 50, max_pd + 50)
            
        if render:   
            self.render()
