import numpy as np
from matplotlib import pyplot as plt
from simulator.gui.panel_components import View
from simulator.utils.simulation_data import SimulationData
from simulator.utils.readable import seconds_to_dhms

class DashboardView(View):
    """
    A class for displaying simulation data (state, control deltas, autopilot status)
    as text inside a matplotlib subplot.
    """

    def __init__(self, fig: plt.Figure, pos: int | tuple):
        super().__init__(fig, pos, is_3d=False)
        self.ax.axis("off")  # Hide axes
        
        # Create a text object that will be updated
        self.text_obj = self.ax.text(
            0.0, 1.0, "", 
            transform=self.ax.transAxes,
            verticalalignment='top',
            horizontalalignment='left',
            fontfamily='monospace',
            fontsize=10
        )
        self.setup_blit([self.text_obj])

    def update_view(self, data: SimulationData) -> None:
        """
        Updates the text with the latest simulation data.
        """
        if data is None:
            return
            
        state = data.uav_state
        deltas = data.control_deltas
        status = data.autopilot_status
        mc = data.mission_control
        wp = mc.target_waypoint
        
        t_sim_str = seconds_to_dhms(data.t_sim)
        
        text = f"$\\mathbf{{Simulation \\ Status}}$\n"
        text += f"Sim Time: {t_sim_str}  |  Step ($\\Delta t$): {data.dt_sim:.4f} s  |  Iterations: {data.k_sim}\n\n"
        
        text += f"$\\mathbf{{Aircraft \\ State}}$\n"
        text += f"Position (m)    :  $p_n$= {state.pn:>7.2f}  |  $p_e$= {state.pe:>7.2f}  |  $h$= {-state.pd:>7.2f}\n"
        text += f"Attitude (deg)  :  $\\phi$= {np.rad2deg(state.roll):>7.2f}  |  $\\theta$= {np.rad2deg(state.pitch):>7.2f}  |  $\\psi$= {np.rad2deg(state.yaw):>7.2f}\n"
        text += f"Velocity (m/s)  :  $u$= {state.u:>7.2f}  |  $v$= {state.v:>7.2f}  |  $w$= {state.w:>7.2f}\n"
        text += f"Rates (deg/s)   :  $p$= {np.rad2deg(state.p):>7.2f}  |  $q$= {np.rad2deg(state.q):>7.2f}  |  $r$= {np.rad2deg(state.r):>7.2f}\n\n"
        
        text += f"$\\mathbf{{Autopilot \\ & \\ Control}}$\n"
        text += f"Altitude ($h$)    :  {status.altitude:>7.2f} m      (Target: {status.target_altitude:>7.2f} m)      [Error: {status.target_altitude - status.altitude:>6.2f} m]\n"
        text += f"Airspeed ($V_a$)  :  {status.airspeed:>7.2f} m/s    (Target: {status.target_airspeed:>7.2f} m/s)    [Error: {status.target_airspeed - status.airspeed:>6.2f} m/s]\n"
        text += f"Course ($\\chi$)    :  {np.rad2deg(status.course):>7.2f} deg    (Target: {np.rad2deg(status.target_course):>7.2f} deg)    [Error: {np.rad2deg(status.target_course - status.course):>6.2f} deg]\n"
        text += f"Control ($\\delta$) :  $\\delta_t$= {deltas.delta_t * 100.0:>5.1f}%  |  $\\delta_a$= {np.rad2deg(deltas.delta_a):>5.2f}$^\\circ$  |  $\\delta_e$= {np.rad2deg(deltas.delta_e):>5.2f}$^\\circ$  |  $\\delta_r$= {np.rad2deg(deltas.delta_r):>5.2f}$^\\circ$\n\n"
        
        text += f"$\\mathbf{{Mission \\ Status}}$\n"
        text += f"Status: {mc.status.upper()}  |  Orbit: {mc.is_on_wait_orbit}  |  Target WP ID: {wp.id if wp else 'None'}\n"
        if wp:
            text += f"WP Target: $p_n$={wp.pn:.1f}, $p_e$={wp.pe:.1f}, $h$={wp.h:.1f}  |  Action: {wp.action_code}\n"
        
        self.text_obj.set_text(text)
        self.render()
