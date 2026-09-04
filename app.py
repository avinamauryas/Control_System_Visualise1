import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
st.sidebar.caption("💡 Recommended range: 0.1 to 50")

if "poles" not in st.session_state:
  st.session_state.poles = [-1.0 + 0j, -2.0 + 0j]
if "zeros" not in st.session_state:
  st.session_state.zeros = []

st.sidebar.subheader("Add Pole/Zero")
r_input = st.sidebar.number_input("Real (sigma)", value=-1.5)
st.sidebar.caption("💡 Keep negative for stability")
im_input = st.sidebar.number_input("Imag (omega)", value=0.0)
st.sidebar.caption("💡 Range: -50 to +50")

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

num_poly = (
    np.real_if_close(np.poly(z_coeffs) * gain_k)
    if len(z_coeffs) > 0
    else np.array([gain_k])
)
den_poly = (
    np.real_if_close(np.poly(p_coeffs)) if len(p_coeffs) > 0 else np.array([1.0])
)
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
  magnitude = np.real(np.log10(1.0 + np.abs(H_s)))
  magnitude = np.clip(
      np.nan_to_num(magnitude, nan=0.0, posinf=10.0, neginf=0.0), 0, 15
  )
  phase = np.real(np.degrees(np.angle(H_s)))
  phase = np.clip(
      np.nan_to_num(phase, nan=0.0, posinf=180.0, neginf=-180.0), -180, 180
  )

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
  mag = np.clip(
      np.real(np.nan_to_num(mag, nan=0.0, posinf=100.0, neginf=-100.0)),
      -200,
      200,
  )
  phase = np.clip(
      np.real(np.nan_to_num(phase, nan=0.0, posinf=360.0, neginf=-360.0)),
      -360,
      360,
  )

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
    fig_bode = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1
    )
    fig_bode.add_trace(
        go.Scatter(
            x=w,
            y=mag,
            mode="lines",
            name="Mag (dB)",
            line=dict(color="blue", width=2),
        ),
        row=1,
        col=1,
    )
    fig_bode.add_trace(
        go.Scatter(
            x=w,
            y=phase,
            mode="lines",
            name="Phase (deg)",
            line=dict(color="red", width=2),
        ),
        row=2,
        col=1,
    )

    fig_bode.update_xaxes(type="log", title_text="Frequency (rad/s)", row=2, col=1)
    fig_bode.update_yaxes(title_text="Magnitude (dB)", row=1, col=1)
    fig_bode.update_yaxes(title_text="Phase (deg)", row=2, col=1)
    fig_bode.update_layout(
        height=420, margin=dict(l=0, r=0, b=0, t=10), showlegend=False
    )
    st.plotly_chart(fig_bode, use_container_width=True)

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
      cleaned = np.real(np.nan_to_num(y_sine, nan=0.0))
      Y_GRID[i, :] = np.clip(cleaned, -50, 50)

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

    cleaned = np.real(np.nan_to_num(y_out, nan=0.0))
    y_out = np.clip(cleaned, -50, 50)
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

# --- TAB 4: 2D TIME DOMAIN ---
with tab4:
  t_2d = np.linspace(0, 10, 200)
  _, y_step = signal.step(sys, T=t_2d)
  _, y_imp = signal.impulse(sys, T=t_2d)
  y_ramp = np.cumsum(y_step) * (t_2d[1] - t_2d[0])
  u_sine = np.sin(1.0 * t_2d)
  _, y_sine, _ = signal.lsim(sys, u_sine, t_2d)

  y_step = np.clip(np.real(np.nan_to_num(y_step, nan=0.0)), -50, 50)
  y_imp = np.clip(np.real(np.nan_to_num(y_imp, nan=0.0)), -50, 50)
  y_ramp = np.clip(np.real(np.nan_to_num(y_ramp, nan=0.0)), -50, 50)
  y_sine = np.clip(np.real(np.nan_to_num(y_sine, nan=0.0)), -50, 50)

  fig_2d = make_subplots(
      rows=2,
      cols=2,
      subplot_titles=(
          "Step Response",
          "Impulse Response",
          "Ramp Response",
          "Sine Response (w=1)",
      ),
  )

  fig_2d.add_trace(
      go.Scatter(
          x=t_2d,
          y=y_step,
          mode="lines",
          line=dict(color="#1f77b4", width=2),
      ),
      row=1,
      col=1,
  )
  fig_2d.add_trace(
      go.Scatter(
          x=t_2d,
          y=y_imp,
          mode="lines",
          line=dict(color="#ff7f0e", width=2),
      ),
      row=1,
      col=2,
  )
  fig_2d.add_trace(
      go.Scatter(
          x=t_2d,
          y=y_ramp,
          mode="lines",
          line=dict(color="#2ca02c", width=2),
      ),
      row=2,
      col=1,
  )
  fig_2d.add_trace(
      go.Scatter(
          x=t_2d,
          y=y_sine,
          mode="lines",
          name="Out",
          line=dict(color="#d62728", width=2),
      ),
      row=2,
      col=2,
  )
  fig_2d.add_trace(
      go.Scatter(
          x=t_2d,
          y=u_sine,
          mode="lines",
          name="In",
          line=dict(color="gray", dash="dash", width=1.5),
      ),
      row=2,
      col=2,
  )

  fig_2d.update_layout(
      height=450, margin=dict(l=0, r=0, b=0, t=30), showlegend=False
  )
  st.plotly_chart(fig_2d, use_container_width=True)
    
