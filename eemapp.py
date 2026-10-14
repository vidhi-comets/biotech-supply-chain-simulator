"""Biotech Supply Chain Under Siege - interactive simulator.
Run:  pip install streamlit numpy pandas plotly
      streamlit run app.py
All cost/price/probability inputs are ASSUMPTIONS - replace with sourced values
and log them in your evidence log.
"""
import numpy as np, pandas as pd, streamlit as st, plotly.graph_objects as go

st.set_page_config(page_title="Biotech Supply Chain Under Siege", layout="wide")
st.title("🧬 Biotech Supply Chain Under Siege – Keep Production Alive")

# ------------------------------------------------------------------ inputs
sb = st.sidebar
sb.header("Scenario inputs")
mean = sb.slider("Normal weekly demand (units)", 200, 5000, 1000, 100)
cv = sb.slider("Demand variability (CV %)", 0, 50, 15) / 100
cover0 = sb.slider("Starting inventory (weeks of cover)", 1.0, 12.0, 6.0, 0.5)
lt_n = sb.slider("Normal lead time (weeks)", 1, 8, 4)
lt_d = sb.slider("Disrupted lead time (weeks)", 4, 26, 14)
dis = sb.slider("Disruption starts (week)", 0, 10, 2)
held = sb.slider("Shipments held at customs", 0, 4, 2)
cdelay = sb.slider("Customs hold (extra weeks)", 1, 12, 6)
fsurge = sb.slider("Freight cost multiplier in disruption", 1.0, 5.0, 2.5, 0.1)
H = sb.slider("Horizon (weeks)", 20, 60, 40)
runs = sb.slider("Monte Carlo runs", 100, 2000, 500, 100)
sb.header("Cost assumptions (₹)")
unit = sb.number_input("Unit purchase cost", value=1000)
fbase = sb.number_input("Normal sea freight per unit", value=100)
hold = sb.number_input("Holding cost per unit per week", value=5)
pen = sb.number_input("Shortage penalty per unit short", value=1500)

STRATS = {
    "1. Business as usual": dict(rop=5, q=4, ch=[(1, "sea")], prem=1.0, alt=0, w=1.0),
    "2. Panic buying": dict(rop=16, q=10, ch=[(1, "sea")], prem=1.15, alt=0, w=1.0),
    "3. Air-freight bridge (40%)": dict(rop=8, q=3, ch=[(.4, "air"), (.6, "sea")], prem=1.0, alt=0, w=1.0),
    "4. Staged + dual-source + tiered allocation": dict(
        rop=10, q=3, ch=[(.3, "alt"), (.7, "sea")], prem=1.05, alt=6, w=0.6),
}
P = dict(mean=mean, cv=cv, cover0=cover0, lt_n=lt_n, lt_d=lt_d, dis=dis, held=held,
         cdelay=cdelay, fsurge=fsurge, H=H, unit=unit, fbase=fbase, hold=hold, pen=pen)


# -------------------------------------------------------------- simulation
@st.cache_data(show_spinner=False)
def simulate(p, s, n, seed=42):
    rng = np.random.default_rng(seed)
    H = p["H"]
    INV, SHORT, DEM = np.zeros((n, H)), np.zeros((n, H)), np.zeros((n, H))
    cost = np.zeros((n, 4))  # purchase, freight, holding, penalty
    peak = np.zeros(n)
    for r in range(n):
        inv = p["cover0"] * p["mean"]
        # steady-state pipeline; the first `held` shipments are stuck at customs
        pipe = [[w + (p["cdelay"] if w <= p["held"] else 0), p["mean"], p["unit"]]
                for w in range(1, p["lt_n"] + 1)]
        for t in range(H):
            inv += sum(q for a, q, _ in pipe if a <= t)
            pipe = [x for x in pipe if x[0] > t]
            d = max(0, rng.normal(p["mean"], p["cv"] * p["mean"]))
            served = min(inv, d); inv -= served
            INV[r, t], SHORT[r, t], DEM[r, t] = inv, d - served, d
            on = t >= p["dis"]
            if inv + sum(q for _, q, _ in pipe) < s["rop"] * p["mean"]:
                qty = s["q"] * p["mean"]
                for share, kind in s["ch"]:
                    if kind == "alt" and t < s["alt"]:
                        kind = "sea"  # alternate supplier not yet qualified
                    base = p["lt_d"] if on else p["lt_n"]
                    if kind == "air":
                        lead, fr = 2, p["fbase"] * 4 * (p["fsurge"] ** .5 if on else 1)
                    elif kind == "alt":
                        lead, fr = 8, p["fbase"] * 1.2 * (p["fsurge"] ** .5 if on else 1)
                    else:
                        lead, fr = base, p["fbase"] * (p["fsurge"] if on else 1)
                    pipe.append([t + lead, qty * share, p["unit"]])
                    cost[r, 0] += qty * share * p["unit"] * s["prem"]
                    cost[r, 1] += qty * share * fr
            cost[r, 2] += inv * p["hold"]
            cost[r, 3] += (d - served) * p["pen"] * s["w"]
            peak[r] = max(peak[r], (inv + sum(q for _, q, _ in pipe)) * p["unit"])
    fill = 1 - SHORT.sum(1) / DEM.sum(1)
    return dict(inv=INV, fill=fill, cost=cost, peak=peak, stock=(SHORT.sum(1) > 0))


tab1, tab2, tab3 = st.tabs(["📈 Inventory simulator", "🔥 Risk heat map", "🗳️ Live poll"])

# ---------------------------------------------------------------- tab 1
with tab1:
    c1, c2 = st.columns([1, 2])
    pick = c1.selectbox("Strategy to inspect", list(STRATS))
    res = simulate(P, STRATS[pick], runs)
    gap = lt_d - cover0
    c1.metric("Coverage gap (disrupted lead time − stock)", f"{gap:.1f} wk")
    c1.metric("Mean fill rate", f"{res['fill'].mean():.1%}")
    c1.metric("P(at least one stock-out)", f"{res['stock'].mean():.0%}")
    c1.metric("Total cost (mean)", f"₹{res['cost'].sum(1).mean() / 1e5:,.1f} lakh")
    c1.metric("Peak working capital (mean)", f"₹{res['peak'].mean() / 1e5:,.1f} lakh")
    x = list(range(H))
    p10, p50, p90 = np.percentile(res["inv"], [10, 50, 90], axis=0)
    fig = go.Figure()
    fig.add_scatter(x=x, y=p90, line=dict(width=0), showlegend=False)
    fig.add_scatter(x=x, y=p10, fill="tonexty", name="P10–P90", line=dict(width=0))
    fig.add_scatter(x=x, y=p50, name="Median inventory", line=dict(width=3))
    fig.add_hline(y=0, line_dash="dash")
    fig.add_vline(x=dis, line_dash="dot", annotation_text="Disruption")
    fig.update_layout(title=f"Inventory (units): {pick}", xaxis_title="Week", yaxis_title="Units", height=420)
    c2.plotly_chart(fig, width="stretch")

    st.subheader("Strategy comparison (same random demand)")
    rows = []
    for k, s in STRATS.items():
        r = simulate(P, s, runs)
        rows.append({"Strategy": k, "Fill rate": r["fill"].mean(), "P(stock-out)": r["stock"].mean(),
                     "Total cost (₹ lakh)": r["cost"].sum(1).mean() / 1e5,
                     "Peak working capital (₹ lakh)": r["peak"].mean() / 1e5})
    df = pd.DataFrame(rows)
    st.dataframe(df.style.format({"Fill rate": "{:.1%}", "P(stock-out)": "{:.0%}",
                                  "Total cost (₹ lakh)": "{:,.1f}",
                                  "Peak working capital (₹ lakh)": "{:,.1f}"}), width="stretch")
    sc = go.Figure(go.Scatter(x=df["Peak working capital (₹ lakh)"], y=df["Fill rate"] * 100, mode="markers+text",
                              text=[k[:2] for k in df["Strategy"]], textposition="top center", marker=dict(size=16)))
    sc.update_layout(title="Service vs working capital (goal: top-left)", height=360,
                     xaxis_title="Peak working capital (₹ lakh)", yaxis_title="Fill rate (%)")
    st.plotly_chart(sc, width="stretch")

# ---------------------------------------------------------------- tab 2
with tab2:
    st.caption("Edit likelihood (L) and impact (I) from 1–5. Tick 'Mitigated' to move a risk to its residual score.")
    base = pd.DataFrame({
        "Risk": ["Antibody stock-out", "Enzyme 14-wk lead time", "Plastics held at customs", "Freight cost spike",
                 "Panic buying / working capital", "Hospital service failure", "Cold-chain break",
                 "Alt-supplier quality", "Currency volatility", "Single-source dependency"],
        "L": [4, 5, 4, 5, 3, 3, 3, 3, 4, 4], "I": [5, 4, 3, 3, 4, 5, 4, 4, 2, 4],
        "Mit_L": [3, 4, 2, 4, 2, 2, 2, 2, 3, 3], "Mit_I": [4, 3, 3, 2, 3, 4, 3, 3, 2, 3],
        "Mitigated": False})
    ed = st.data_editor(base, width="stretch", hide_index=True,
                        column_config={c: st.column_config.NumberColumn(min_value=1, max_value=5, step=1)
                                       for c in ["L", "I", "Mit_L", "Mit_I"]})
    ed["l"] = np.where(ed["Mitigated"], ed["Mit_L"], ed["L"])
    ed["i"] = np.where(ed["Mitigated"], ed["Mit_I"], ed["I"])
    ed["score"] = ed["l"] * ed["i"]
    z = np.array([[l * i for l in range(1, 6)] for i in range(1, 6)])
    hm = go.Figure(go.Heatmap(z=z, x=list(range(1, 6)), y=list(range(1, 6)), zmin=1, zmax=25,
                              colorscale=[[0, "#6fcf97"], [.3, "#f2e27a"], [.55, "#f2a65a"], [1, "#e5534b"]], showscale=False, opacity=.9))
    seen = {}
    for k, r in ed.iterrows():
        key = (r.l, r.i); n = seen.get(key, 0); seen[key] = n + 1
        hm.add_scatter(x=[r.l + (n % 2 - .5) * .4], y=[r.i + (n // 2 - .3) * .3], mode="markers+text",
                       text=[f"R{k + 1}"], marker=dict(size=26, color="white", line=dict(width=2, color="black")),
                       textfont=dict(color="black"), hovertext=f"{r.Risk}: score {r.score}", showlegend=False)
    hm.update_layout(height=520, xaxis_title="Likelihood", yaxis_title="Impact")
    a, b = st.columns([2, 1])
    a.plotly_chart(hm, width="stretch")
    b.metric("Total exposure (Σ L×I)", int(ed.score.sum()))
    b.metric("Critical risks (≥15)", int((ed.score >= 15).sum()))
    b.dataframe(ed[["Risk", "score"]].sort_values("score", ascending=False), hide_index=True)


# ---------------------------------------------------------------- tab 3
@st.cache_resource
def store():  # shared by every browser session on this server
    return {"q1": {}, "q2": {}, "q3": {}}


QS = {"q1": ("Which risk should be tackled FIRST?",
             ["R1 Antibody stock-out", "R2 Enzyme lead time", "R6 Hospital service failure", "R10 Single-source"]),
      "q2": ("Best strategy for the next 14 weeks?", list(STRATS)),
      "q3": ("Extra safety stock you would fund?", ["2 weeks", "4 weeks", "6 weeks", "8 weeks"])}
with tab3:
    st.caption("Share your network URL (shown in the terminal) with the class. Results refresh every 3 seconds.")
    name = st.text_input("Your name / team", key="who")
    with st.form("poll"):
        ans = {k: st.radio(q, o, index=None, key=k + "r") for k, (q, o) in QS.items()}
        if st.form_submit_button("Submit vote") and name:
            for k, v in ans.items():
                if v:
                    store()[k][name] = v
            st.success("Vote recorded")
        elif name == "":
            pass
    if st.button("Clear all votes"):
        for k in store(): store()[k].clear()

    @st.fragment(run_every=3)
    def results():
        for k, (q, o) in QS.items():
            counts = pd.Series(store()[k]).value_counts().reindex(o, fill_value=0)
            st.write(f"**{q}** ({counts.sum()} votes)")
            st.bar_chart(counts)
    results()
