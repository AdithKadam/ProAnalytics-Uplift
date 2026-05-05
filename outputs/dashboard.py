import json
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ProAnalytics · Uplift Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f !important;
    color: #e8e6f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse 80% 60% at 50% -10%, rgba(108,92,231,0.18) 0%, transparent 70%),
                radial-gradient(ellipse 50% 40% at 90% 80%, rgba(0,210,200,0.08) 0%, transparent 60%),
                #0a0a0f !important;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { background: #0d0d14 !important; }
.block-container { padding: 1.5rem 2.5rem 3rem !important; max-width: 1400px !important; }

h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

hr { border: none !important; border-top: 1px solid rgba(255,255,255,0.06) !important; margin: 1.5rem 0 !important; }

[data-testid="stTabs"] [role="tablist"] {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    gap: 2px !important;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    color: rgba(232,230,240,0.45) !important;
    border-radius: 8px !important;
    padding: 8px 18px !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.01em !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: rgba(108,92,231,0.25) !important;
    color: #c4b5fd !important;
    border-bottom: none !important;
}
[data-testid="stTabs"] [role="tab"]:hover {
    color: #e8e6f0 !important;
    background: rgba(255,255,255,0.05) !important;
}

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 14px !important;
    padding: 20px 22px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stMetric"]:hover { border-color: rgba(108,92,231,0.35) !important; }
[data-testid="stMetricLabel"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: rgba(232,230,240,0.4) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 26px !important;
    font-weight: 700 !important;
    color: #e8e6f0 !important;
}
[data-testid="stMetricDelta"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
}

[data-testid="stSelectbox"] > div > div,
[data-testid="stTextInput"] > div > div > input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #e8e6f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: rgba(108,92,231,0.4); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ───────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#9d9bb5", size=12),
    margin=dict(t=44, b=28, l=12, r=12),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.08)",
        tickcolor="rgba(255,255,255,0.08)", tickfont=dict(family="DM Mono", size=10),
        title_font=dict(family="DM Sans", size=11, color="#6c6a82"),
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.08)",
        tickcolor="rgba(255,255,255,0.08)", tickfont=dict(family="DM Mono", size=10),
        title_font=dict(family="DM Sans", size=11, color="#6c6a82"),
    ),
    legend=dict(
        bgcolor="rgba(255,255,255,0.04)", bordercolor="rgba(255,255,255,0.08)",
        borderwidth=1, font=dict(family="DM Sans", size=11),
    ),
    hoverlabel=dict(
        bgcolor="#1a1a2e", bordercolor="rgba(108,92,231,0.5)",
        font=dict(family="DM Mono", size=11),
    ),
)

def apply_layout(fig, title="", height=320, **kwargs):
    layout = {**PLOT_LAYOUT, "height": height, **kwargs}
    if title:
        layout["title"] = dict(
            text=title,
            font=dict(family="Syne", size=14, color="#c4b5fd"),
            x=0.0, xanchor="left", pad=dict(l=4)
        )
    fig.update_layout(**layout)
    return fig

# ── Color palette ──────────────────────────────────────────────────────────────
C = {
    "purple": "#6c5ce7", "teal": "#00d2c8", "amber": "#f5c842",
    "red": "#f5605a", "green": "#5af5a0",
    "purple_l": "#c4b5fd", "teal_l": "#99f6f2",
}
SEG_COLORS = {
    "Persuadable":          C["purple"],
    "Moderate Responder":   C["teal"],
    "Sure Thing / Neutral": C["amber"],
    "Do-Not-Disturb":       C["red"],
}

# ── Load data ──────────────────────────────────────────────────────────────────
# Resolve path relative to this file so it works from any working directory
_DATA_FILE = Path(__file__).parent / "dashboard_data.json"

@st.cache_data
def load_data():
    if not _DATA_FILE.exists():
        st.error(
            "**dashboard_data.json not found.**\n\n"
            "Please run the pipeline first to generate the data:\n"
            "```\npython uplift_pipeline.py\n```\n"
            "Then refresh this page."
        )
        st.stop()
    with open(_DATA_FILE) as f:
        return json.load(f)

DATA = load_data()
kpis = DATA["kpis"]
segs = DATA["segments"]
tc   = DATA["treatment_control"]
meta = DATA["meta"]

# Pre-compute header meta to avoid backslash-in-f-string (Python < 3.12)
_meta_labels = [
    f"TEAM · {meta['team'].upper()}",
    f"DATASET · {meta['dataset'].upper()}",
    f"N · {meta['n_customers']:,} CUSTOMERS",
    f"MODELS · {' · '.join(m.upper() for m in meta['models_used'])}",
]
_span = '<span style="font-family:\'DM Mono\',monospace;font-size:11px;color:rgba(232,230,240,0.35);letter-spacing:0.06em;">'
header_meta_html = "".join(f"{_span}{l}</span>" for l in _meta_labels)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="padding:2rem 0 1.5rem;border-bottom:1px solid rgba(255,255,255,0.06);margin-bottom:1.8rem;">
  <div style="display:flex;align-items:baseline;gap:14px;margin-bottom:8px;">
    <span style="font-family:'Syne',sans-serif;font-size:32px;font-weight:800;color:#e8e6f0;letter-spacing:-0.02em;">
      Promotion Uplift Intelligence
    </span>
    <span style="font-family:'DM Mono',monospace;font-size:11px;color:{C['purple_l']};
                 background:rgba(108,92,231,0.15);border:1px solid rgba(108,92,231,0.3);
                 border-radius:6px;padding:3px 10px;letter-spacing:0.08em;">LIVE</span>
  </div>
  <div style="display:flex;gap:24px;flex-wrap:wrap;">
    {header_meta_html}
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tabs = st.tabs(["◈  Overview", "⬡  Segments", "◎  Model Performance",
                "◆  ROI Simulator",
                # "⊞  Customer Explorer", 
                "●  Cluster Analysis"])


# ══════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════
with tabs[0]:

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Persuadable Customers", f"{kpis['persuadable_count']:,}", f"{kpis['persuadable_pct']}% of base")
    c2.metric("Campaign Lift", f"+{kpis['lift']}pp", f"Treated {kpis['treatment_purchase_rate']}%  ·  Control {kpis['control_purchase_rate']}%")
    c3.metric("Avg Uplift Score", f"{kpis['avg_uplift_score']:.4f}", "X-Learner estimate")
    c4.metric("Max Net ROI", f"${kpis['max_net_roi']:,.0f}", f"Top {kpis['optimal_targeting_pct']}%  ·  {DATA['best_roi']['roi_pct']}% return")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        hist_df = pd.DataFrame(DATA["uplift_histogram"])
        fig = go.Figure(go.Bar(
            x=hist_df["bin"], y=hist_df["count"],
            marker=dict(
                color=hist_df["bin"],
                colorscale=[[0, C["red"]], [0.45, "#3d3860"], [0.55, "#3d5560"], [1, C["teal"]]],
                line=dict(width=0),
            ),
            hovertemplate="<b>Score:</b> %{x:.3f}<br><b>Count:</b> %{y}<extra></extra>"
        ))
        fig.add_vline(x=0, line_color="rgba(255,255,255,0.2)", line_dash="dot", line_width=1)
        apply_layout(fig, "Uplift Score Distribution", height=300,
                     xaxis=dict(**PLOT_LAYOUT["xaxis"], title="Uplift Score"),
                     yaxis=dict(**PLOT_LAYOUT["yaxis"], title="Customers"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col2:
        fig2 = go.Figure(go.Pie(
            labels=list(segs.keys()), values=list(segs.values()), hole=0.65,
            marker=dict(colors=[SEG_COLORS[s] for s in segs], line=dict(color="#0a0a0f", width=3)),
            textinfo="percent", textfont=dict(family="DM Mono", size=11, color="#0a0a0f"),
            hovertemplate="<b>%{label}</b><br>%{value:,} customers  ·  %{percent}<extra></extra>"
        ))
        fig2.add_annotation(text=f"{sum(segs.values()):,}<br>CUSTOMERS",
                            x=0.5, y=0.5, showarrow=False,
                            font=dict(family="Syne", color="#e8e6f0", size=16))
        apply_layout(fig2, "Customer Segment Breakdown", height=300, showlegend=True,
                     legend=dict(**PLOT_LAYOUT["legend"], orientation="v", x=1.02, y=0.5))
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    col3, col4 = st.columns(2)
    with col3:
        uc = pd.DataFrame(DATA["uplift_curve"])
        random_line = [kpis["lift"] * p / 100 for p in uc["percentile"]]
        fig3 = go.Figure()
        fig3.add_traces([
            go.Scatter(x=uc["percentile"], y=random_line, name="Random",
                       line=dict(color="rgba(255,255,255,0.15)", dash="dot", width=1.5)),
            go.Scatter(x=uc["percentile"], y=uc["model"], name="X-Learner",
                       line=dict(color=C["purple"], width=2.5),
                       fill="tonexty", fillcolor="rgba(108,92,231,0.08)"),
        ])
        apply_layout(fig3, "Uplift Curve vs Random", height=280,
                     xaxis=dict(**PLOT_LAYOUT["xaxis"], title="% Targeted"),
                     yaxis=dict(**PLOT_LAYOUT["yaxis"], title="Cumulative Lift (pp)"))
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    with col4:
        qc = pd.DataFrame(DATA["qini_curve"])
        fig4 = go.Figure()
        fig4.add_traces([
            go.Scatter(x=qc["percentile"], y=qc["random"], name="Random",
                       line=dict(color="rgba(255,255,255,0.15)", dash="dot", width=1.5)),
            go.Scatter(x=qc["percentile"], y=qc["model"], name="Qini",
                       line=dict(color=C["teal"], width=2.5),
                       fill="tonexty", fillcolor="rgba(0,210,200,0.08)"),
        ])
        apply_layout(fig4, f"Qini Curve  ·  coeff = {kpis['qini_coefficient']}", height=280,
                     xaxis=dict(**PLOT_LAYOUT["xaxis"], title="% Targeted"),
                     yaxis=dict(**PLOT_LAYOUT["yaxis"], title="Qini Score"))
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

    st.divider()
    st.markdown('<p style="font-family:\'Syne\',sans-serif;font-weight:700;font-size:15px;color:#e8e6f0;margin-bottom:12px;">Feature Importance</p>', unsafe_allow_html=True)
    fi = pd.DataFrame(DATA["feature_importance"]).sort_values("importance")
    fig5 = go.Figure(go.Bar(
        x=fi["importance"], y=fi["feature"], orientation="h",
        marker=dict(color=fi["importance"],
                    colorscale=[[0, "rgba(108,92,231,0.3)"], [1, C["purple_l"]]],
                    line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
    ))
    apply_layout(fig5, height=360,
                 xaxis=dict(**PLOT_LAYOUT["xaxis"], title="Importance Score"),
                 yaxis={**PLOT_LAYOUT["yaxis"], "tickfont": dict(family="DM Mono", size=10)})
    st.plotly_chart(fig5, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════
# TAB 2 — SEGMENTS
# ══════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<p style="font-family:\'Syne\',sans-serif;font-weight:700;font-size:22px;color:#e8e6f0;margin-bottom:16px;">Customer Segments</p>', unsafe_allow_html=True)

    total = kpis["total_customers"]
    seg_descriptions = {
        "Persuadable":          ("High uplift. True targets — promotion causally drives their purchase.", C["purple"]),
        "Moderate Responder":   ("Some uplift. Worth including in broader campaigns, lower priority.", C["teal"]),
        "Sure Thing / Neutral": ("Would buy anyway. Spending budget here reduces overall ROI.", C["amber"]),
        "Do-Not-Disturb":       ("Negative uplift. Promotions actively reduce purchase probability.", C["red"]),
    }
    cols = st.columns(4)
    for i, (seg, cnt) in enumerate(segs.items()):
        desc, color = seg_descriptions[seg]
        pct = cnt / total * 100
        cols[i].markdown(f"""
        <div style="background:rgba(255,255,255,0.02);border:1px solid {color}33;border-top:3px solid {color};
                    border-radius:12px;padding:18px 16px;height:100%;">
          <div style="font-family:'DM Mono',monospace;font-size:9px;letter-spacing:0.12em;
                      color:{color};text-transform:uppercase;margin-bottom:8px;">{seg}</div>
          <div style="font-family:'Syne',sans-serif;font-size:28px;font-weight:800;color:#e8e6f0;">{cnt:,}</div>
          <div style="font-family:'DM Mono',monospace;font-size:11px;color:rgba(232,230,240,0.4);
                      margin-bottom:10px;">{pct:.1f}% of base</div>
          <div style="font-size:11px;color:rgba(232,230,240,0.5);line-height:1.5;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        pca_df_seg = pd.DataFrame(DATA["pca_clusters"])
        seg_means = pca_df_seg.groupby("segment")["uplift"].mean().to_dict()
        seg_means = {s: seg_means.get(s, 0.0) for s in SEG_COLORS}
        fig = go.Figure(go.Bar(
            x=list(seg_means.keys()), y=list(seg_means.values()),
            marker=dict(color=[SEG_COLORS[s] for s in seg_means], opacity=0.85, line=dict(width=0)),
            hovertemplate="<b>%{x}</b><br>Avg Uplift: %{y:.3f}<extra></extra>"
        ))
        apply_layout(fig, "Average Uplift by Segment", height=320,
                     yaxis=dict(**PLOT_LAYOUT["yaxis"], title="Avg Uplift Score",
                                zeroline=True, zerolinecolor="rgba(255,255,255,0.12)"),
                     xaxis={**PLOT_LAYOUT["xaxis"], "tickfont": dict(family="DM Sans", size=11)})
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col2:
        metrics = ["Purchase Rate (%)", "Avg Revenue ($)"]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Treatment", x=metrics,
                              y=[tc["treatment_purchase_rate"], tc["avg_revenue_treated"]],
                              marker=dict(color=C["purple"], opacity=0.85, line=dict(width=0))))
        fig2.add_trace(go.Bar(name="Control", x=metrics,
                              y=[tc["control_purchase_rate"], tc["avg_revenue_control"]],
                              marker=dict(color=C["red"], opacity=0.7, line=dict(width=0))))
        apply_layout(fig2, "Treatment vs Control", height=320, barmode="group")
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    st.markdown(f"""
    <div style="background:rgba(108,92,231,0.08);border:1px solid rgba(108,92,231,0.2);
                border-left:3px solid {C['purple']};border-radius:10px;padding:16px 20px;margin-top:8px;">
      <span style="font-family:'Syne',sans-serif;font-weight:700;color:{C['purple_l']};font-size:13px;">KEY INSIGHT</span>
      <p style="margin:6px 0 0;font-size:13px;color:rgba(232,230,240,0.7);line-height:1.6;">
        By targeting only <b style="color:{C['purple']}">Persuadable</b> customers, you avoid wasting discounts on
        <b style="color:{C['amber']}">Sure Thing</b> customers who buy regardless, and
        <b style="color:{C['red']}">Do-Not-Disturb</b> customers whose purchase probability actively decreases with promotion exposure.
      </p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<p style="font-family:\'Syne\',sans-serif;font-weight:700;font-size:22px;color:#e8e6f0;margin-bottom:16px;">Model Performance</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown('<p style="font-family:\'DM Mono\',monospace;font-size:10px;letter-spacing:0.1em;color:rgba(232,230,240,0.4);text-transform:uppercase;margin-bottom:12px;">Baseline AUC Scores</p>', unsafe_allow_html=True)
        bm = DATA["baseline_models"]
        auc_colors = {"Logistic Regression": C["teal"], "Random Forest": C["purple"],
                      "XGBoost": C["amber"], "Gradient Boosting": C["green"]}
        for name, v in bm.items():
            color = auc_colors.get(name, C["purple"])
            st.markdown(f"""
            <div style="margin-bottom:12px;">
              <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
                <span style="font-size:12px;color:rgba(232,230,240,0.7);">{name}</span>
                <span style="font-family:'DM Mono',monospace;font-size:12px;color:{color};font-weight:500;">{v['auc']:.4f}</span>
              </div>
              <div style="background:rgba(255,255,255,0.05);border-radius:4px;height:5px;">
                <div style="background:{color};border-radius:4px;height:5px;width:{v['auc']*100}%;opacity:0.8;"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<p style="font-family:\'DM Mono\',monospace;font-size:10px;letter-spacing:0.1em;color:rgba(232,230,240,0.4);text-transform:uppercase;margin:20px 0 12px;">Treatment vs Control</p>', unsafe_allow_html=True)
        for label, t_val, c_val in [
            ("Customers",     f"{tc['n_treated']:,}",             f"{tc['n_control']:,}"),
            ("Purchase Rate", f"{tc['treatment_purchase_rate']}%", f"{tc['control_purchase_rate']}%"),
            ("Avg Revenue",   f"${tc['avg_revenue_treated']}",     f"${tc['avg_revenue_control']}"),
        ]:
            st.markdown(f"""
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin-bottom:8px;
                        background:rgba(255,255,255,0.02);border-radius:8px;padding:10px 12px;">
              <span style="font-family:'DM Mono',monospace;font-size:10px;color:rgba(232,230,240,0.35);text-transform:uppercase;letter-spacing:0.06em;">{label}</span>
              <span style="font-family:'DM Mono',monospace;font-size:12px;color:{C['purple_l']};text-align:center;">{t_val}</span>
              <span style="font-family:'DM Mono',monospace;font-size:12px;color:{C['red']};text-align:center;">{c_val}</span>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        uc = pd.DataFrame(DATA["uplift_curve"])
        random_line = [kpis["lift"] * p / 100 for p in uc["percentile"]]
        fig = go.Figure()
        fig.add_traces([
            go.Scatter(x=uc["percentile"], y=random_line, name="Random",
                       line=dict(color="rgba(255,255,255,0.15)", dash="dot", width=1.5)),
            go.Scatter(x=uc["percentile"], y=uc["model"], name="X-Learner",
                       line=dict(color=C["purple"], width=2.5),
                       fill="tonexty", fillcolor="rgba(108,92,231,0.1)"),
        ])
        apply_layout(fig, "Uplift Curve — X-Learner vs Random", height=300,
                     xaxis=dict(**PLOT_LAYOUT["xaxis"], title="% Targeted"),
                     yaxis=dict(**PLOT_LAYOUT["yaxis"], title="Cumulative Lift (pp)"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    fi = pd.DataFrame(DATA["feature_importance"]).sort_values("importance")
    fig2 = go.Figure(go.Bar(
        x=fi["importance"], y=fi["feature"], orientation="h",
        marker=dict(color=fi["importance"],
                    colorscale=[[0, "rgba(0,210,200,0.2)"], [1, C["teal"]]],
                    line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>%{x:.4f}<extra></extra>"
    ))
    apply_layout(fig2, "Feature Importance Ranking", height=360,
                 xaxis=dict(**PLOT_LAYOUT["xaxis"], title="Importance"),
                 yaxis={**PLOT_LAYOUT["yaxis"], "tickfont": dict(family="DM Mono", size=10)})
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════
# TAB 4 — ROI SIMULATOR
# ══════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<p style="font-family:\'Syne\',sans-serif;font-weight:700;font-size:22px;color:#e8e6f0;margin-bottom:4px;">ROI Simulator</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-family:\'DM Sans\',sans-serif;font-size:13px;color:rgba(232,230,240,0.4);margin-bottom:20px;">Drag the slider to simulate net ROI at different targeting thresholds</p>', unsafe_allow_html=True)

    roi_df = pd.DataFrame(DATA["roi_simulation"])
    optimal_pct = int(DATA["best_roi"]["pct_targeted"])

    pct = st.slider("Target top % of customers by uplift score",
                    min_value=5, max_value=100, step=5, value=optimal_pct, format="%d%%")

    row = roi_df[roi_df["pct_targeted"] == pct].iloc[0]
    is_optimal = int(row["pct_targeted"]) == optimal_pct

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Customers Targeted", f"{int(row['n_customers']):,}")
    c2.metric("Incremental Revenue", f"${row['incremental_revenue']:,.0f}")
    c3.metric("Discount Cost", f"${row['discount_cost']:,.0f}")
    c4.metric("Net ROI", f"${row['net_roi']:,.0f}")
    c5.metric("ROI %", f"{row['roi_pct']:.1f}%", "✦ Optimal" if is_optimal else None)

    if is_optimal:
        st.markdown(f"""
        <div style="background:rgba(90,245,160,0.06);border:1px solid rgba(90,245,160,0.2);
                    border-radius:10px;padding:10px 16px;margin:8px 0 16px;display:inline-block;">
          <span style="font-family:'DM Mono',monospace;font-size:11px;color:{C['green']};letter-spacing:0.08em;">
            ✦ OPTIMAL TARGETING THRESHOLD
          </span>
        </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        pos_df = roi_df[roi_df["net_roi"] >= 0]
        if not pos_df.empty:
            fig.add_trace(go.Scatter(x=pos_df["pct_targeted"], y=pos_df["net_roi"],
                                     fill="tozeroy", fillcolor="rgba(90,245,160,0.07)",
                                     line=dict(color=C["green"], width=0), showlegend=False))
        fig.add_trace(go.Scatter(
            x=roi_df["pct_targeted"], y=roi_df["net_roi"],
            mode="lines+markers", name="Net ROI",
            line=dict(color=C["green"], width=2.5), marker=dict(size=5, color=C["green"]),
            hovertemplate="%{x}% targeted<br><b>$%{y:,.0f}</b><extra></extra>"
        ))
        fig.add_vline(x=pct, line_color=C["amber"], line_dash="dash", line_width=1.5,
                      annotation=dict(text=f" {pct}%", font=dict(color=C["amber"], family="DM Mono", size=11)))
        fig.add_vline(x=optimal_pct, line_color=C["green"], line_dash="dot", line_width=1,
                      annotation=dict(text=" Optimal", font=dict(color=C["green"], family="DM Mono", size=10)))
        fig.add_hline(y=0, line_color="rgba(255,255,255,0.12)", line_width=1)
        apply_layout(fig, "Net ROI by Targeting %", height=320,
                     xaxis=dict(**PLOT_LAYOUT["xaxis"], title="% Targeted"),
                     yaxis=dict(**PLOT_LAYOUT["yaxis"], title="Net ROI ($)"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Incremental Revenue", x=roi_df["pct_targeted"], y=roi_df["incremental_revenue"],
                              marker=dict(color=C["green"], opacity=0.75, line=dict(width=0))))
        fig2.add_trace(go.Bar(name="Discount Cost", x=roi_df["pct_targeted"], y=roi_df["discount_cost"],
                              marker=dict(color=C["red"], opacity=0.6, line=dict(width=0))))
        apply_layout(fig2, "Revenue vs Cost", height=320, barmode="group",
                     xaxis=dict(**PLOT_LAYOUT["xaxis"], title="% Targeted"),
                     yaxis=dict(**PLOT_LAYOUT["yaxis"], title="$"))
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════
# TAB 5 — CUSTOMER EXPLORER
# ══════════════════════════════════════════════════════
# with tabs[4]:
#     st.markdown('<p style="font-family:\'Syne\',sans-serif;font-weight:700;font-size:22px;color:#e8e6f0;margin-bottom:4px;">Customer Explorer</p>', unsafe_allow_html=True)
#     st.markdown('<p style="font-family:\'DM Sans\',sans-serif;font-size:13px;color:rgba(232,230,240,0.4);margin-bottom:16px;">Top 100 customers ranked by X-Learner uplift score</p>', unsafe_allow_html=True)

#     cust_df = pd.DataFrame(DATA["top_customers"])
#     cust_df["customer_id"] = cust_df["customer_id"].astype(str)

#     col1, col2, col3 = st.columns([2, 2, 1])
#     with col1:
#         search = st.text_input("🔍  Search Customer ID", placeholder="e.g. 1551")
#     with col2:
#         seg_filter = st.selectbox("Filter by Segment", ["All"] + list(segs.keys()))
#     with col3:
#         st.metric("Showing", f"{len(cust_df)}")

#     filtered = cust_df.copy()
#     if search:
#         filtered = filtered[filtered["customer_id"].str.contains(search)]
#     if seg_filter != "All":
#         filtered = filtered[filtered["segment"] == seg_filter]

#     display_df = filtered.rename(columns={
#         "customer_id": "Customer ID", "uplift_x": "X-Learner Uplift",
#         "uplift_t": "T-Learner Uplift", "p_treated": "P(Buy|Treated)",
#         "p_control": "P(Buy|Control)", "segment": "Segment",
#         "purchase_frequency": "Purchase Freq", "avg_basket_value": "Avg Basket ($)",
#         "rfm_score": "RFM Score", "promo_sensitivity_score": "Promo Sensitivity",
#         "treatment": "Treated", "purchase": "Purchased",
#     })
#     st.dataframe(
#         display_df.style
#             .background_gradient(subset=["X-Learner Uplift"], cmap="RdYlGn")
#             .set_properties(**{
#                 "background-color": "#0f0f18",
#                 "color": "#e8e6f0",
#                 "border-color": "rgba(255,255,255,0.06)",
#                 "font-family": "DM Mono, monospace",
#                 "font-size": "12px",
#             })
#             .set_table_styles([
#                 {"selector": "th", "props": [
#                     ("background-color", "#16162a"),
#                     ("color", "rgba(232,230,240,0.5)"),
#                     ("font-family", "DM Mono, monospace"),
#                     ("font-size", "10px"),
#                     ("letter-spacing", "0.06em"),
#                     ("text-transform", "uppercase"),
#                     ("border-bottom", "1px solid rgba(255,255,255,0.08)"),
#                     ("padding", "10px 12px"),
#                 ]},
#                 {"selector": "td", "props": [
#                     ("padding", "8px 12px"),
#                     ("border-bottom", "1px solid rgba(255,255,255,0.04)"),
#                 ]},
#                 {"selector": "tr:hover td", "props": [
#                     ("background-color", "rgba(108,92,231,0.08)"),
#                 ]},
#             ])
#             .format({"X-Learner Uplift": "{:.4f}", "T-Learner Uplift": "{:.4f}",
#                      "P(Buy|Treated)": "{:.3f}", "P(Buy|Control)": "{:.3f}",
#                      "Avg Basket ($)": "{:.2f}", "RFM Score": "{:.4f}",
#                      "Promo Sensitivity": "{:.4f}"}),
#         use_container_width=True, height=500
#     )


# ══════════════════════════════════════════════════════
# TAB 6 — CLUSTER ANALYSIS
# ══════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<p style="font-family:\'Syne\',sans-serif;font-weight:700;font-size:22px;color:#e8e6f0;margin-bottom:4px;">Cluster Analysis  ·  PCA</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-family:\'DM Sans\',sans-serif;font-size:13px;color:rgba(232,230,240,0.4);margin-bottom:16px;">K-Means behavioural clusters projected onto 2D via PCA</p>', unsafe_allow_html=True)

    pca_df = pd.DataFrame(DATA["pca_clusters"])
    pca_df["cluster"] = pca_df["cluster"].astype(str)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(pca_df, x="x", y="y", color="cluster",
                         color_discrete_map={"0": C["purple"], "1": C["teal"], "2": C["amber"], "3": C["red"]},
                         labels={"x": "PC1", "y": "PC2", "cluster": "Cluster"},
                         hover_data={"segment": True, "uplift": ":.4f", "x": False, "y": False})
        fig.update_traces(marker=dict(size=4.5, opacity=0.65, line=dict(width=0)))
        apply_layout(fig, "Behavioral Clusters (K-Means)", height=400,
                     xaxis=dict(**PLOT_LAYOUT["xaxis"], title="PC1"),
                     yaxis=dict(**PLOT_LAYOUT["yaxis"], title="PC2"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col2:
        fig2 = px.scatter(pca_df, x="x", y="y", color="segment",
                          color_discrete_map=SEG_COLORS,
                          labels={"x": "PC1", "y": "PC2", "segment": "Segment"},
                          hover_data={"cluster": True, "uplift": ":.4f", "x": False, "y": False})
        fig2.update_traces(marker=dict(size=4.5, opacity=0.65, line=dict(width=0)))
        apply_layout(fig2, "Uplift Segments in PCA Space", height=400,
                     xaxis=dict(**PLOT_LAYOUT["xaxis"], title="PC1"),
                     yaxis=dict(**PLOT_LAYOUT["yaxis"], title="PC2"))
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # st.divider()
    # st.markdown('<p style="font-family:\'DM Mono\',monospace;font-size:10px;letter-spacing:0.1em;color:rgba(232,230,240,0.4);text-transform:uppercase;margin-bottom:12px;">Cluster Summary</p>', unsafe_allow_html=True)

    # cluster_summary = pca_df.groupby("cluster").agg(
    #     Count=("uplift", "count"),
    #     Avg_Uplift=("uplift", "mean"),
    #     Dominant_Segment=("segment", lambda x: x.value_counts().index[0])
    # ).reset_index()
    # cluster_summary.columns = ["Cluster", "Count", "Avg Uplift", "Dominant Segment"]
    # cluster_summary["Avg Uplift"] = cluster_summary["Avg Uplift"].round(4)
    # st.dataframe(
    #     cluster_summary.style
    #         .background_gradient(subset=["Avg Uplift"], cmap="RdYlGn")
    #         .set_properties(**{
    #             "background-color": "#0f0f18",
    #             "color": "#e8e6f0",
    #             "font-family": "DM Mono, monospace",
    #             "font-size": "12px",
    #         })
    #         .set_table_styles([
    #             {"selector": "th", "props": [
    #                 ("background-color", "#16162a"),
    #                 ("color", "rgba(232,230,240,0.5)"),
    #                 ("font-family", "DM Mono, monospace"),
    #                 ("font-size", "10px"),
    #                 ("letter-spacing", "0.06em"),
    #                 ("text-transform", "uppercase"),
    #                 ("border-bottom", "1px solid rgba(255,255,255,0.08)"),
    #                 ("padding", "10px 12px"),
    #             ]},
    #             {"selector": "td", "props": [
    #                 ("padding", "8px 12px"),
    #                 ("border-bottom", "1px solid rgba(255,255,255,0.04)"),
    #             ]},
    #         ])
    #         .format({"Avg Uplift": "{:.4f}"}),
    #     use_container_width=True, height=220
    # )