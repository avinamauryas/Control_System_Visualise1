import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy import signal
import streamlit as st

# Set up page layout
st.set_page_config(
    page_title="Control Systems Analyzer", layout="wide", page_icon="🎛️"
)

st.title("🎛️ Advanced Control Systems: Laplace, Bode & Time Domain Analyzer")
st.markdown(
    "Analyze your poles, zeros, stability, and interactive 3D/2D responses."
)

# --- SIDEBAR: Controls ---
st.sidebar.header("System Parameter Controls")

# Gain K
gain_k = st.sidebar.number_input("System Gain (K)", value=1.0, step=0.1)

# Poles & Zeros configuration using session state
if "poles" not in st.session_state:
  st.session_state.poles = [-1.0 + 0j, -2.0 + 0j]
if "zeros" not in st.session_state:
  st.session_state.zeros = []

st.sidebar.subheader("Add Pole or Zero")
r_input = st.sidebar.number_input("Real Part (σ)", value=-1.5)
im_input = st.sidebar.number_input("Imaginary Part (ω)", value=0.0)
item_type = st.sidebar.radio("Type", ["Pole", "Zero"])

col1, col2 = st.sidebar.columns(2)
if col1.button("Add"):
  val = complex(r_input, im_input)
  if item_type == "Pole":
    st.session_state.poles.append(val)
  else:
    st.session_state.zeros.append(val)
  st.rerun()

if col2.button("Clear All"):
  st.session_state.poles = []
  st.session_state.zeros = []
  st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Active System Elements")
st.sidebar.write(f"**Poles (x):** {[str(p) for p in st.session_state.poles]}")
st.sidebar.write(f"**Zeros (o):** {[str(z) for z in st.session_state.zeros]}")

# --- MAIN WINDOW TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "3D Laplace Surface",
    "Bode Plot & Stability",
    "3D Time Domain Response",
    "2D Time Responses",
])

# Prepare system coefficients
z_coeffs = st.session_state.zeros
p_coeffs = st.session_state.poles
num_poly = np.poly(z_coeffs) * gain_k if len(z_coeffs) > 0 else np.array([gain_k])
den_poly = np.poly(p_coeffs) if len(p_coeffs) > 0 else np.array([1.0])
sys = signal.TransferFunction(num_poly, den_poly)

# --- TAB 1: 3D LAPLACE (Plotly Interactive) ---
with tab1:
  st.subheader("3D Laplace Domain Surfaces (Interactive WebGL)")
  st.markdown(
      "Click and drag to rotate, scroll to zoom in/out on the 3D surfaces."
  )

  sigma = np.linspace(-4, 4, 50)
  omega = np.linspace(-10, 10, 50)
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
  phase = np.degrees(np.angle(H_s))

  col_l1, col_l2 = st.columns(2)

  with col_l1:
    fig_mag = go.Figure(
        data=[
            go.Surface(
                z=magnitude, x=sigma, y=omega, colorscale="Viridis", opacity=0.9
            )
        ]
    )
    fig_mag.update_layout(
        title=f"Magnitude Surface (K={gain_k})",
        scene=dict(
            xaxis_title="σ (Real)",
            yaxis_title="ω (Imag)",
            zaxis_title="log₁₀(1 + |H(s)|)",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=500,
    )
    st.plotly_chart(fig_mag, use_container_width=True)

  with col_l2:
    fig_phase = go.Figure(
        data=[
            go.Surface(
                z=phase, x=sigma, y=omega, colorscale="Plasma", opacity=0.9
            )
        ]
    )
    fig_phase.update_layout(
        title="Phase Surface (°)",
        scene=dict(
            xaxis_title="σ (Real)",
            yaxis_title="ω (Imag)",
            zaxis_title="∠H(s) (°)",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=500,
    )
    st.plotly_chart(fig_phase, use_container_width=True)

# --- TAB 2: BODE & STABILITY ---
with tab2:
  st.subheader("Bode Frequency Response & Margins")
  w = np.logspace(-2, 3, 1000)
  w, mag, phase = signal.bode(sys, w)

  # Calculate Margins
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

  fig_bode, (ax_bm, ax_bp) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
  ax_bm.semilogx(w, mag, color="b", lw=2)
  ax_bm.axhline(0, color="gray", linestyle="--")
  ax_bm.set_ylabel("Magnitude (dB)")
  ax_bm.grid(True, which="both")

  ax_bp.semilogx(w, phase, color="r", lw=2)
  ax_bp.axhline(-180, color="gray", linestyle="--")
  ax_bp.set_xlabel("Frequency (rad/s)")
  ax_bp.set_ylabel("Phase (deg)")
  ax_bp.grid(True, which="both")

  st.pyplot(fig_bode)

  st.info(
      f"**Stability Analysis Status:** {'🟢 STABLE' if is_stable else '🔴 UNSTABLE'}"
      f"  \n• Gain Crossover Frequency ($\omega_{{gc}}$): {omega_gc:.2f} rad/s"
      f"  \n• Phase Crossover Frequency ($\omega_{{pc}}$): {omega_pc:.2f} rad/s"
      f"  \n• Phase Margin (PM): {pm:.2f}° | Gain Margin (GM): {gm_dB:.2f} dB"
  )

# --- TAB 3: 3D TIME DOMAIN (Plotly Interactive) ---
with tab3:
  st.subheader("3D Time Domain Response Visualizer")
  input_choice = st.selectbox(
      "Select Input Type", ["Step", "Impulse", "Ramp", "Sine"]
  )

  t = np.linspace(0, 10, 150)

  if input_choice == "Sine":
    freqs = np.linspace(0.5, 5.0, 20)
    T_GRID, F_GRID = np.meshgrid(t, freqs)
    Y_GRID = np.zeros_like(T_GRID)
    for i, f in enumerate(freqs):
      u = np.sin(2 * np.pi * f * t)
      _, y_sine, _ = signal.lsim(sys, u, t)
      Y_GRID[i, :] = y_sine

    fig_t3d = go.Figure(
        data=[
            go.Surface(
                z=Y_GRID, x=t, y=freqs, colorscale="Coolwarm", opacity=0.9
            )
        ]
    )
    fig_t3d.update_layout(
        scene=dict(
            xaxis_title="Time (s)",
            yaxis_title="Input Frequency (Hz)",
            zaxis_title="Output Amplitude",
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=600,
    )
  else:
    if input_choice == "Step":
      _, y_out = signal.step(sys, T=t)
    elif input_choice == "Impulse":
      _, y_out = signal.impulse(sys, T=t)
    elif input_choice == "Ramp":
      _, y_step = signal.step(sys, T=t)
      y_out = np.cumsum(y_step) * (t[1] - t[0])

    depth = np.linspace(0, 1, 10)
    T_GRID, D_GRID = np.meshgrid(t, depth)
    Y_GRID = np.tile(y_out, (len(depth), 1))

    fig_t3d = go.Figure(
        data=[
            go.Surface(
                z=Y_GRID, x=t, y=depth, colorscale="Viridis", opacity=0.9
            )
        ]
    )
    fig_t3d.update_layout(
        scene=dict(
            xaxis_title="Time (s)",
            yaxis_title="Profile Width",
            zaxis_title="Output Amplitude",
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=600,
    )

  st.plotly_chart(fig_t3d, use_container_width=True)

# --- TAB 4: 2D TIME DOMAIN GRID ---
with tab4:
  st.subheader("Comprehensive 2D Time Responses")
  t_2d = np.linspace(0, 10, 300)

  _, y_step = signal.step(sys, T=t_2d)
  _, y_imp = signal.impulse(sys, T=t_2d)
  y_ramp = np.cumsum(y_step) * (t_2d[1] - t_2d[0])
  u_sine = np.sin(1.0 * t_2d)
  _, y_sine, _ = signal.lsim(sys, u_sine, t_2d)

  fig_2d, axs = plt.subplots(2, 2, figsize=(12, 8))

  axs[0, 0].plot(t_2d, y_step, color="tab:blue", lw=2)
  axs[0, 0].set_title("Step Response")
  axs[0, 0].grid(True)

  axs[0, 1].plot(t_2d, y_imp, color="tab:orange", lw=2)
  axs[0, 1].set_title("Impulse Response")
  axs[0, 1].grid(True)

  axs[1, 0].plot(t_2d, y_ramp, color="tab:green", lw=2)
  axs[1, 0].set_title("Ramp Response")
  axs[1, 0].grid(True)

  axs[1, 1].plot(t_2d, y_sine, color="tab:red", lw=2, label="Output")
  axs[1, 1].plot(
      t_2d, u_sine, color="gray", linestyle="--", alpha=0.7, label="Input"
  )
  axs[1, 1].set_title("Sine Response (ω = 1 rad/s)")
  axs[1, 1].legend()
  axs[1, 1].grid(True)

  plt.tight_layout()
  st.pyplot(fig_2d)
