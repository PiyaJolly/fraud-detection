import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

st.set_page_config(
    page_title="Fraud Detection: Cost-Based Decisions",
    page_icon="💳",
    layout="wide",
)

# ---------- Global CSS: readable text, sensible content width ----------
st.markdown(
    """
    <style>
      .block-container { max-width: 1500px; padding-top: 2.5rem; padding-bottom: 3rem;
                         padding-left: 3rem; padding-right: 3rem; }
      /* Base + body text larger and more readable */
      html, body, [class*="css"] { font-size: 17.5px; }
      .stMarkdown p, .stMarkdown li { font-size: 1.15rem; line-height: 1.7; color: #cbd5e1; }
      /* Section headers */
      h2 { font-size: 2rem !important; font-weight: 700 !important; margin-top: 0.6rem !important; }
      h3, h4 { color: #e5e7eb !important; font-size: 1.35rem !important; }
      /* Metric labels and values */
      [data-testid="stMetricValue"] { font-size: 2.3rem !important; }
      [data-testid="stMetricLabel"] { font-size: 1rem !important; color: #94a3b8 !important; }
      /* Info / success / warning boxes text */
      .stAlert p { font-size: 1.12rem !important; }
      /* Slider labels and buttons */
      .stSlider label p { font-size: 1.05rem !important; }
      .stButton button { font-size: 1.05rem !important; padding: 0.55rem 1rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Dark chart styling ----------
VIOLET = "#a78bfa"
PINK = "#f472b6"
MUTED = "#64748b"

plt.rcParams.update({
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "figure.dpi": 130,
    "text.color": "#e5e7eb",
    "axes.labelcolor": "#cbd5e1",
    "axes.titlecolor": "#f1f5f9",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
    "axes.edgecolor": "#334155",
    "grid.color": "#1f2937",
    "font.size": 9,
})


# ---------- Load saved artifacts ----------
@st.cache_resource
def load_model():
    return joblib.load("app_artifacts/xgb_model.pkl")


@st.cache_data
def load_demo():
    return pd.read_csv("app_artifacts/demo_data.csv")


model = load_model()
demo = load_demo()

y_true = demo["Class"].values
y_prob = demo["fraud_prob"].values
feature_cols = [c for c in demo.columns if c not in ("Class", "fraud_prob")]
total_frauds = int(y_true.sum())


# ---------- Header ----------
st.markdown(
    """
    <div style="padding: 2px 0;">
      <h1 style="margin:0; font-size:2.4rem; font-weight:800;
                 background: linear-gradient(90deg,#a78bfa,#f472b6);
                 -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        Credit Card Fraud Detection
      </h1>
      <p style="margin:6px 0 0 0; color:#94a3b8; font-size:1.15rem;">
        Choosing the right <em>decision</em>, not just the right model
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<p style='color:#cbd5e1; font-size:1.22rem; line-height:1.7; max-width:960px;'>"
    "Fraud is rare (under 0.2% of transactions), so a model can be 99.8% accurate while catching "
    "almost no fraud. The interesting question isn't accuracy, it's: <b style='color:#e5e7eb;'>what "
    "does each type of mistake cost, and where should we draw the line?</b> This tool lets you set "
    "those costs and see the best decision update live.</p>",
    unsafe_allow_html=True,
)

st.divider()


# ---------- 1. The cost slider (hero feature) ----------
st.header("1. Set the cost of each mistake")

c1, c2 = st.columns(2)
cost_missed = c1.slider(
    "Cost of a missed fraud ($)",
    min_value=10, max_value=2000, value=150, step=10,
    help="What the business loses when a fraudulent transaction slips through.",
)
cost_false = c2.slider(
    "Cost of a false alarm ($)",
    min_value=1, max_value=100, value=5, step=1,
    help="The cost of reviewing / the friction of wrongly flagging a legitimate customer.",
)

thresholds = np.linspace(0.01, 0.99, 99)
costs = []
for t in thresholds:
    preds = (y_prob >= t).astype(int)
    cm = confusion_matrix(y_true, preds)
    costs.append(cm[1][0] * cost_missed + cm[0][1] * cost_false)
costs = np.array(costs)

best_idx = int(costs.argmin())
best_t = thresholds[best_idx]
best_cost = costs[best_idx]

cm_best = confusion_matrix(y_true, (y_prob >= best_t).astype(int))
caught, missed, false_alarms = int(cm_best[1][1]), int(cm_best[1][0]), int(cm_best[0][1])
recall = caught / total_frauds * 100

left, right = st.columns([3, 2], gap="large")

with left:
    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    ax.plot(thresholds, costs, color=VIOLET, linewidth=2.2)
    ax.axvline(best_t, color=PINK, linestyle="--", linewidth=1.8, label=f"Optimal = {best_t:.2f}")
    ax.axvline(0.5, color=MUTED, linestyle=":", linewidth=1.3, label="Default = 0.50")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Total cost ($)")
    ax.set_title("Total business cost at every threshold", pad=10, fontweight="bold", fontsize=10)
    ax.legend(facecolor="#151a28", edgecolor="#334155", labelcolor="#e5e7eb", fontsize=8)
    ax.grid(alpha=0.35)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

with right:
    st.markdown("#### At the cost-optimal threshold")
    st.metric("Optimal threshold", f"{best_t:.2f}")
    m1, m2 = st.columns(2)
    m1.metric("Frauds caught", f"{caught} / {total_frauds}", f"{recall:.0f}% recall")
    m2.metric("Frauds missed", missed)
    m3, m4 = st.columns(2)
    m3.metric("False alarms", f"{false_alarms:,}")
    m4.metric("Total cost", f"${best_cost:,.0f}")

st.info(
    "Move the sliders and watch the optimal threshold shift. When missed fraud is expensive, the "
    "line moves left (catch more, tolerate more false alarms). When false alarms hurt more, it "
    "moves right. The model never changes, only the decision does."
)

st.divider()


# ---------- 2. Score a transaction ----------
st.header("2. Score a real transaction")
st.markdown(
    "Pick a transaction from the held-out test set and see the model's fraud probability and the "
    "decision it would make at your current threshold."
)

if "picked" not in st.session_state:
    st.session_state.picked = demo.sample(1, random_state=1).index[0]

b1, b2, b3 = st.columns(3)
if b1.button("Random legitimate transaction", use_container_width=True):
    st.session_state.picked = demo[demo["Class"] == 0].sample(1).index[0]
if b2.button("Random fraudulent transaction", use_container_width=True):
    st.session_state.picked = demo[demo["Class"] == 1].sample(1).index[0]
if b3.button("Random transaction (either)", use_container_width=True):
    st.session_state.picked = demo.sample(1).index[0]

row = demo.loc[st.session_state.picked]
prob = row["fraud_prob"]
actual = "Fraud" if row["Class"] == 1 else "Legitimate"
decision = "FLAG as fraud" if prob >= best_t else "Allow"

r1, r2, r3 = st.columns(3)
r1.metric("Model's fraud probability", f"{prob*100:.1f}%")
r2.metric(f"Decision at threshold {best_t:.2f}", decision)
r3.metric("Actually was", actual)

if decision == "FLAG as fraud" and actual == "Fraud":
    st.success("Correct catch: the model flagged real fraud.")
elif decision == "Allow" and actual == "Legitimate":
    st.success("Correct: a legitimate transaction allowed through.")
elif decision == "FLAG as fraud" and actual == "Legitimate":
    st.warning("False alarm: a legitimate customer would be wrongly flagged here.")
else:
    st.error("Missed fraud: this real fraud would slip through at the current threshold.")

st.divider()


# ---------- 3. Feature importance ----------
st.header("3. What drives the model's decisions?")
st.markdown(
    "The features `V1`-`V28` are anonymised (PCA-transformed to protect privacy), which is realistic "
    "since real fraud data is almost always de-identified. We can't name them individually, but we "
    "can still see which carry the most weight. A model you can interpret is one a business can trust."
)

names = getattr(model, "feature_names_in_", np.array(feature_cols))
importances = model.feature_importances_
imp = pd.DataFrame({"feature": names, "importance": importances})
imp = imp.sort_values("importance", ascending=False).head(10)

_, mid, _ = st.columns([1, 8, 1])
with mid:
    fig2, ax2 = plt.subplots(figsize=(7, 3.1))
    ax2.barh(imp["feature"][::-1], imp["importance"][::-1], color=VIOLET)
    ax2.set_xlabel("Importance")
    ax2.set_title("Top 10 features driving fraud predictions", pad=10, fontweight="bold", fontsize=10)
    ax2.grid(alpha=0.35, axis="x")
    for s in ["top", "right"]:
        ax2.spines[s].set_visible(False)
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)

st.divider()


# ---------- 4. Recommendation ----------
st.header("4. Recommendation")

st.markdown(
    f"""
**For the cost assumptions currently set** (a missed fraud costs **\\${cost_missed}**, a false
alarm costs **\\${cost_false}**):

- Deploy the XGBoost model with a decision threshold of **{best_t:.2f}**, not the default 0.50.
- At this operating point the model catches **{caught} of {total_frauds} frauds ({recall:.0f}%)**,
  misses **{missed}**, and raises **{false_alarms:,} false alarms**, for a total modelled cost of
  **\\${best_cost:,.0f}** on the test set.

**The key insight:** the model is fixed, but the decision threshold is a business lever. If the
business treats missed fraud as more costly, the optimal threshold moves lower (catch more, accept
more false alarms). If customer friction is the greater concern, it moves higher. This choice
should be made deliberately, with the cost of each error in mind, rather than left at an arbitrary
default.
"""
)

st.caption(
    "Cost figures are illustrative and adjustable above. Built with a gradient-boosted model on "
    "the ULB credit card fraud dataset. The full analysis lives in the accompanying notebook."
)