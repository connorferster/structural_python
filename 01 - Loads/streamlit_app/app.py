import json
import streamlit as st
import matplotlib.pyplot as plt
import pycba as cba
import loadfactors as lf
import numpy as np

occupancies_dict = {
    "Residential": lf.Load(D=1.2, L=2.4, S=0., W=0., E=0.),
    "Office": lf.Load(D=3.6, L=2.4, S=0., W=0., E=0.),
    "Roof": lf.Load(D=0.9, L=1.0, S=1.4, W=-1.2, E=0.),
    "Amenity Roof": lf.Load(D=0.9, L=4.8, S=1.4, W=-1.2, E=0.),
    "Exterior Cladding": lf.Load(D=1.8, L=0., S=0., W=0., E=0.),
    "Rooftop Mech": lf.Load(D=8.5, L=0., S=0., W=0., E=0.),
}

with open("NBCC_vec_full.json") as file:
    load_combinations = json.load(file)

def analyze_beam(
    span_left,
    cant_right,
    w1,
    w2,
    p1,
    a1,
    p2,
    a2,
    EI: float = 1.
) -> dict[str, cba.results.BeamResults]:
    r = [-1, 0, -1, 0, 0, 0] # Three nodes: two vertical supports, one free end
    beam_results = {}
    for load_type in w1._fields: # load_type = D, L, S, etc.
        # To keep each load_type separate, we need to build a separate analysis
        # beam for each load type and then store the analysis results in a dict
        # organized by load_type.
        udl_span_left = [1, 1, getattr(w1, load_type), 0, 0]
        udl_span_right = [2, 1, getattr(w2, load_type), span_left, span_left + cant_right]
        pl_span_left = [1, 2, getattr(p1, load_type), a1, 0]
        pl_span_right = [2, 2, getattr(p2, load_type), a2, 0]
        lm = [
            udl_span_left,
            udl_span_right,
            pl_span_left,
            pl_span_right,
        ]
        beam = cba.BeamAnalysis([span_left, cant_right], EI, r, lm)
        beam.analyze()
        beam_results.update({load_type: beam.beam_results})
    return beam_results

st.write("# Factoring loads by occupancies")

st.image("BeamDiagram.png")

st.sidebar.write("### Spans")
L_backspan = st.sidebar.slider("Backspan length", min_value=3, max_value=20)
L_cantilever = st.sidebar.slider("Cantilever length", min_value=3, max_value=20)

st.sidebar.write("### Loading")


w1_occ = st.sidebar.selectbox("W1 Occupancy", occupancies_dict.keys())
w1_load = occupancies_dict[w1_occ]
st.sidebar.write(w1_load)
w1_trib = st.sidebar.number_input("W1 Tributary area")

st.sidebar.write("---")

w2_occ = st.sidebar.selectbox("W2 Occupancy", occupancies_dict.keys())
w2_load = occupancies_dict[w2_occ]
st.sidebar.write(w2_load)
w2_trib = st.sidebar.number_input("W2 Tributary area")
st.sidebar.write("---")

p1_occ = st.sidebar.selectbox("P1 Occupancy", occupancies_dict.keys())
p1_load = occupancies_dict[p1_occ]
st.sidebar.write(p1_load)
p1_trib = st.sidebar.number_input("P1 Tributary area")
a1_loc = st.sidebar.slider("a1 Location:", max_value = L_backspan + L_cantilever)
st.sidebar.write("---")

p2_occ = st.sidebar.selectbox("P2 Occupancy", occupancies_dict.keys())
p2_load = occupancies_dict[p2_occ]
st.sidebar.write(p2_load)
p2_trib = st.sidebar.number_input("P2 Tributary area")
a2_loc = st.sidebar.slider("a2 Location:", max_value = L_backspan + L_cantilever)
st.sidebar.write("---")

st.sidebar.image("logo-black.png")

beam_results = analyze_beam(
    L_backspan, 
    L_cantilever,
    lf.Load(*np.array(w1_load) * w1_trib),
    lf.Load(*np.array(w2_load) * w2_trib),
    lf.Load(*np.array(p1_load) * p1_trib),
    a1_loc,
    lf.Load(*np.array(p2_load) * p2_trib),
    a2_loc,
    )

moment_results = lf.Load(*[
    [beam_results["D"].results.M],
    [beam_results['L'].results.M],
    [beam_results['S'].results.M],
    [beam_results['W'].results.M],
    [beam_results['E'].results.M],
])

shear_results = lf.Load(*[
    [beam_results["D"].results.V],
    [beam_results['L'].results.V],
    [beam_results['S'].results.V],
    [beam_results['W'].results.V],
    [beam_results['E'].results.V],
])

r0_results = lf.Load(*[
    [beam_results["D"].R[0]],
    [beam_results['L'].R[0]],
    [beam_results['S'].R[0]],
    [beam_results['W'].R[0]],
    [beam_results['E'].R[0]],
])

r1_results = lf.Load(*[
    [beam_results["D"].R[1]],
    [beam_results['L'].R[1]],
    [beam_results['S'].R[1]],
    [beam_results['W'].R[1]],
    [beam_results['E'].R[1]],
])

moment_envelope_max = lf.factored_max(moment_results, load_combinations)
moment_envelope_min = lf.factored_min(moment_results, load_combinations)

shear_envelope_max = lf.factored_max(shear_results, load_combinations)
shear_envelope_min = lf.factored_min(shear_results, load_combinations)

r0_max = lf.factored_max(r0_results, load_combinations)
r1_max = lf.factored_max(r1_results, load_combinations)

r0_min = lf.factored_min(r0_results, load_combinations)
r1_min = lf.factored_min(r1_results, load_combinations)

st.write("### Factored reactions")
st.table(
    {"R1": {"Max": r0_max, "Min": r0_min},
    "R2": {"Max": r1_max, "Min": r1_min}}
)

st.write("### Factored Moment Diagram")
lc_select = st.selectbox("Plot factored load case: ", load_combinations.keys())
lc_combo = load_combinations[lc_select]
moment_lc = lf.factor_load(np.array(moment_results).T, np.array(lc_combo))
shear_lc = lf.factor_load(np.array(shear_results).T, np.array(lc_combo))

fig, ax = plt.subplots(figsize=(8, 2))

x_data = np.linspace(0, L_backspan + L_cantilever, 206)
beam_line, = ax.plot(x_data, np.zeros(206), 'k')
beam_line.set_label("Beam")
max_moment_line, = ax.plot(x_data, moment_envelope_max, 'r')
max_moment_line.set_label("Max moment envelope")
min_moment_line, = ax.plot(x_data, moment_envelope_min, 'b')
min_moment_line.set_label("Min moment envelope")
lc_moment_line, = ax.plot(x_data, moment_lc, 'g')
lc_moment_line.set_label(lc_select)
ax.set_xlim((-1, L_backspan + L_cantilever + 1))
ax.legend(loc='lower right', bbox_to_anchor=(1.4, 0.5))


st.pyplot(fig)

fig2, ax2 = plt.subplots(figsize=(8, 2))


x_data = np.linspace(0, L_backspan + L_cantilever, 206)
beam_line, = ax2.plot(x_data, np.zeros(206), 'k')
beam_line.set_label("Beam")
max_shear_line, = ax2.plot(x_data, shear_envelope_max, 'r')
max_shear_line.set_label("Max shear envelope")
min_shear_line, = ax2.plot(x_data, shear_envelope_min, 'b')
min_shear_line.set_label("Min shear envelope")
lc_shear_line, = ax2.plot(x_data, shear_lc, 'g')
lc_shear_line.set_label(lc_select)
ax2.set_aspect('auto')
ax2.set_xlim((-1, L_backspan + L_cantilever + 1))
ax2.legend(loc='lower right', bbox_to_anchor=(1.4, 0.5))

st.write("### Factored Shear Diagram")
st.pyplot(fig2)