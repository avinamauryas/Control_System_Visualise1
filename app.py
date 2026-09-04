import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy import signal
import streamlit as st

# Set up page layout to wide
st.set_page_config(
    page_title="Control Systems Lab", layout="wide", page_icon="⚡"
)

st.title("⚡ Advanced Control Systems Analyzer")

# --- SIDEBAR: Controls ---
st.sidebar.header("System Controls")
gain_k = st.sidebar.number_input(
    "Gain (K)", value=1.0, step=0.1, key="system_gain_k"
)

if "poles" not in st.session_state:
  st.session_state.poles = [-1.0 + 0j, -2.0 + 0j]
if "zeros" not in st.session_state:
  st.session_state.zeros = []

st.sidebar.subheader("Add Pole/Zero")
r_input = st.sidebar.number_input("Real (sigma)", value=-1.5)
im_input = st.sidebar.number_input("Imag (omega)", value=0.0)
item_type = st.sidebar.radio("Type", ["Pole", "Zero"], horizontal=True)

col_b1, col_b2 = st.sidebar.columns(2)
if col_b1.button("Add"):
  val = complex(r_input, im_input)
  if item_type == "Pole":
    st.session_state.poles.append(val)
  else:
    st.session_state.zeros.append(val)
  st.rerun()

if col_b2.button("Clear"):
  st.session_state.poles = []
  st.session_state.zeros = []
  st.rerun()

st.sidebar.markdown("---")
st.sidebar.text(
    f"Poles: {[str(p) for p in st.session_state.poles]}\nZeros:"
    f" {[str(z) for z in st.session_state.zeros]}"
)

# --- MAIN WINDOW TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "3D Laplace",
    "Bode & Stability",
    "3D Time Domain",
    "2D Time Responses",
])

z_coeffs = st.session_state.zeros
p_coeffs = st.session_state.poles
num_poly = np.poly(z_coeffs) * gain_k if len(z_coeffs) > 0 else np.array([gain_k])
den_poly = np.poly(p_coeffs) if len(p_coeffs) > 0 else np.array([1.0])
sys = signal.TransferFunction(num_poly, den_poly)

# --- TAB 1: 3D LAPLACE ---
with tab1:
  sigma = np.linspace(-4, 4, 40)
  omega = np.linspace(-10, 10, 40)
  SIGMA, OMEGA = np.meshgrid(sigma, omega)
  s = SIGMA + 1j * OMEGA

  num = complex(gain_k, 0)
  for z in z_coeffs:
    num = num * (s - z)
  den = 1.0
  for p in p_coeffs:
    den = den * (s - p)

  H_s = num / den
  magnitude = np.log10(1.0 + np.abs(H_s))
  magnitude = np.nan_to_num(magnitude, nan=0.0, posinf=10.0, neginf=0.0)
  phase = np.degrees(np.angle(H_s))
  phase = np.nan_to_num(phase, nan=0.0, posinf=180.0, neginf=-180.0)

  col_l1, col_l2 = st.columns(2)
  with col_l1:
    fig_mag = go.Figure(
        data=[
            go.Surface(
                z=magnitude.tolist(),
                x=sigma.tolist(),
                y=omega.tolist(),
                colorscale="Viridis",
                opacity=0.9,
            )
        ]
    )
    fig_mag.update_layout(
        title=f"Magnitude (K={gain_k})",
        scene=dict(
            xaxis_title="sigma", yaxis_title="omega", zaxis_title="log10(1+|H|)"
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=400,
    )
    st.plotly_chart(fig_mag, use_container_width=True)

  with col_l2:
    fig_phase = go.Figure(
        data=[
            go.Surface(
                z=phase.tolist(),
                x=sigma.tolist(),
                y=omega.tolist(),
                colorscale="Plasma",
                opacity=0.9,
            )
        ]
    )
    fig_phase.update_layout(
        title="Phase Surface (deg)",
        scene=dict(xaxis_title="sigma", yaxis_title="omega", zaxis_title="Angle"),
        margin=dict(l=0, r=0, b=0, t=30),
        height=400,
    )
    st.plotly_chart(fig_phase, use_container_width=True)

# --- TAB 2: BODE & STABILITY ---
with tab2:
  w = np.logspace(-2, 3, 500)
  w, mag, phase = signal.bode(sys, w)
  mag = np.nan_to_num(mag, nan=0.0, posinf=100.0, neginf=-100.0)
  phase = np.nan_to_num(phase, nan=0.0, posinf=360.0, neginf=-360.0)

  zero_dB = np.where(np.diff(np.sign(mag)))[0]
  omega_gc = w[zero_dB[0]] if len(zero_dB) > 0 else np.nan
  phase_180 = np.where(np.diff(np.sign(phase + 180)))[0]
  omega_pc = w[phase_180[0]] if len(phase_180) > 0 else np.nan
  pm = (
      180 + phase[np.argmin(np.abs(w - omega_gc))]
      if not np.isnan(omega_gc)
      else np.nan
  )
  gm_dB = (
      -mag[np.argmin(np.abs(w - omega_pc))]
      if not np.isnan(omega_pc)
      else np.inf
  )
  is_stable = (pm > 0 if not np.isnan(pm) else True) and (
      gm_dB > 0 if not np.isinf(gm_dB) else True
  )

  col_b1, col_b2 = st.columns([2, 1])
  with col_b1:
    fig_bode, (ax_bm, ax_bp) = plt.subplots(2, 1, figsize=(6, 4.5), sharex=True)
    ax_bm.semilogx(w, mag, color="b", lw=1.5)
    ax_bm.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax_bm.set_ylabel("Mag (dB)", fontsize=9)
    ax_bm.grid(True, which="both")
    ax_bm.tick_params(labelsize=8)

    ax_bp.semilogx(w, phase, color="r", lw=1.5)
    ax_bp.axhline(-180, color="gray", linestyle="--", alpha=0.5)
    ax_bp.set_xlabel("Freq (rad/s)", fontsize=9)
    ax_bp.set_ylabel("Phase (deg)", fontsize=9)
    ax_bp.grid(True, which="both")
    ax_bp.tick_params(labelsize=8)

    plt.tight_layout(pad=1.0)
    st.pyplot(fig_bode)

  with col_b2:
    st.markdown("### Stability")
    status_color = "STABLE" if is_stable else "UNSTABLE"
    st.markdown(f"**Status:** {status_color}")
    st.markdown(f"• **w_gc:** {omega_gc:.2f} rad/s")
    st.markdown(f"• **w_pc:** {omega_pc:.2f} rad/s")
    st.markdown(f"• **Phase Margin:** {pm:.2f} deg")
    st.markdown(f"• **Gain Margin:** {gm_dB:.2f} dB")

# --- TAB 3: 3D TIME DOMAIN ---
with tab3:
  input_choice = st.selectbox(
      "Input Type", ["Step", "Impulse", "Ramp", "Sine"]
  )
  t = np.linspace(0, 10, 100)

  if input_choice == "Sine":
    freqs = np.linspace(0.5, 5.0, 15)
    Y_GRID = np.zeros((len(freqs), len(t)))
    for i, f in enumerate(freqs):
      u = np.sin(2 * np.pi * f * t)
      _, y_sine, _ = signal.lsim(sys, u, t)
      Y_GRID[i, :] = np.clip(np.nan_to_num(y_sine, nan=0.0), -50, 50)

    fig_t3d = go.Figure(
        data=[
            go.Surface(
                z=Y_GRID.tolist(),
                x=t.tolist(),
                y=freqs.tolist(),
                colorscale="Jet",
                opacity=0.9,
            )
        ]
    )
    fig_t3d.update_layout(
        scene=dict(xaxis_title="Time", yaxis_title="Freq", zaxis_title="Output"),
        margin=dict(l=0, r=0, b=0, t=10),
        height=420,
    )
  else:
    if input_choice == "Step":
      _, y_out = signal.step(sys, T=t)
    elif input_choice == "Impulse":
      _, y_out = signal.impulse(sys, T=t)
    elif input_choice == "Ramp":
      _, y_step = signal.step(sys, T=t)
      y_out = np.cumsum(y_step) * (t[1] - t[0])

    y_out = np.clip(np.nan_to_num(y_out, nan=0.0), -50, 50)
    depth = np.linspace(0, 1, 8)
    Y_GRID = np.tile(y_out, (len(depth), 1))

    fig_t3d = go.Figure(
        data=[
            go.Surface(
                z=Y_GRID.tolist(),
                x=t.tolist(),
                y=depth.tolist(),
                colorscale="Viridis",
                opacity=0.9,
            )
        ]
    )
    fig_t3d.update_layout(
        scene=dict(xaxis_title="Time", yaxis_title="Width", zaxis_title="Output"),
        margin=dict(l=0, r=0, b=0, t=10),
        height=420,
    )

  st.plotly_chart(fig_t3d, use_container_width=True)

# --- TAB 4: 2D TIME DOMAIN ---
with tab4:
  t_2d = np.linspace(0, 10, 200)
  _, y_step = signal.step(sys, T=t_2d)
  _, y_imp = signal.impulse(sys, T=t_2d)
  y_ramp = np.cumsum(y_step) * (t_2d[1] - t_2d[0])
  u_sine = np.sin(1.0 * t_2d)
  _, y_sine, _ = signal.lsim(sys, u_sine, t_2d)

  y_step = np.clip(np.nan_to_num(y_step, nan=0.0), -50, 50)
  y_imp = np.clip(np.nan_to_num(y_imp, nan=0.0), -50, 50)
  y_ramp = np.clip(np.nan_to_num(y_ramp, nan=0.0), -50, 50)
  y_sine = np.clip(np.nan_to_num(y_sine, nan=0.0), -50, 50)

  fig_2d, axs = plt.subplots(2, 2, figsize=(10, 4.5))

  axs[0, 0].plot(t_2d, y_step, color="tab:blue", lw=1.5)
  axs[0, 0].set_title("Step", fontsize=9)
  axs[0, 0].grid(True)
  axs[0, 0].tick_params(labelsize=8)

  axs[0, 1].plot(t_2d, y_imp, color="tab:orange", lw=1.5)
  axs[0, 1].set_title("Impulse", fontsize=9)
  axs[0, 1].grid(True)
  axs[0, 1].tick_params(labelsize=8)

  axs[1, 0].plot(t_2d, y_ramp, color="tab:green", lw=1.5)
  axs[1, 0].set_title("Ramp", fontsize=9)
  axs[1, 0].grid(True)
  axs[1, 0].tick_params(labelsize=8)

  axs[1, 1].plot(t_2d, y_sine, color="tab:red", lw=1.5, label="Out")
  axs[1, 1].plot(
      t_2d, u_sine, color="gray", linestyle="--", alpha=0.7, label="In"
  )
  axs[1, 1].set_title("Sine (w=1)", fontsize=9)
  axs[1, 1].legend(fontsize=7)
  axs[1, 1].grid(True)
  axs[1, 1].tick_params(labelsize=8)

  plt.tight_layout(pad=1.0)
  st.pyplot(fig_2d)
