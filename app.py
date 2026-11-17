"""Streamlit showcase for the trained spam detection model."""

from __future__ import annotations

import re
import warnings
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "spam_detection_pipeline.joblib"
DATA_PATH = BASE_DIR / "spam_data.csv"

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")

EXAMPLES = {
    "Choose an example…": "",
    "Prize claim · obvious spam": (
        "Subject: You are our $5,000 grand-prize winner!\n\n"
        "CONGRATULATIONS! Your email address was selected in our international "
        "rewards drawing. A $5,000 cash prize is waiting for you. To release the "
        "funds, reply with your full name, telephone number, and mailing address, "
        "then pay the small processing fee. You must claim the reward today or it "
        "will be transferred to another winner.\n\nPrize Department"
    ),
    "Payroll verification · phishing": (
        "Subject: Immediate action required — payroll account locked\n\n"
        "Dear employee,\n\nWe detected unusual activity and temporarily suspended "
        "your payroll account. To avoid a delay to your next payment, verify your "
        "identity within 30 minutes using the secure link below. You will need your "
        "employee login, password, and banking information. Failure to complete "
        "verification today may result in permanent account closure.\n\nPayroll Security Team"
    ),
    "Unpaid invoice · malicious attachment": (
        "Subject: Final notice — invoice 88421 remains unpaid\n\n"
        "Our records show that invoice 88421 is now 45 days overdue. Open the "
        "attached payment statement and enable editing to review the outstanding "
        "balance. Remit payment immediately to prevent collection activity and a "
        "suspension of services. If you believe this message was sent in error, "
        "reply with your account credentials so we can investigate.\n\nAccounts Receivable"
    ),
    "Limited-time promotion · marketing": (
        "Subject: Weekend offer — save 30% on your next order\n\n"
        "Hello,\n\nOur weekend sale has started. Use promotional code WEEKEND30 "
        "at checkout to receive 30% off selected products through Sunday at midnight. "
        "This is a promotional message sent because you subscribed to product updates. "
        "You can update your email preferences at any time.\n\nThe Store Team"
    ),
    "Project review · legitimate work email": (
        "Subject: Draft launch plan for Thursday's review\n\n"
        "Hi team,\n\nI incorporated the comments from yesterday's planning session "
        "into the attached launch plan. Please review sections two and four before our "
        "meeting on Thursday at 10:00 a.m. I would especially appreciate feedback on "
        "the proposed timeline and ownership table. Add comments in the document or "
        "send them to me by Wednesday afternoon.\n\nThanks,\nMorgan"
    ),
    "Appointment confirmation · legitimate service email": (
        "Subject: Your service appointment is confirmed\n\n"
        "Hello,\n\nThis message confirms your service appointment for Tuesday, "
        "November 24 at 2:30 p.m. No action is required. If you need to reschedule, "
        "please use the contact information from your original booking confirmation. "
        "Our representative will arrive during the selected appointment window.\n\n"
        "Customer Care"
    ),
}


st.set_page_config(
    page_title="Inbox Shield · Spam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The trained model was not found. Run `python spam_detect.py` first."
        )
    # The artifact may have been produced by a different scikit-learn patch
    # release. This app only reads its learned numeric parameters.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Trying to unpickle estimator")
        return joblib.load(MODEL_PATH)


@st.cache_data
def load_feature_names() -> list[str]:
    """Read only the CSV header; the 31 MB dataset is not loaded into the app."""
    columns = pd.read_csv(DATA_PATH, nrows=0).columns.tolist()
    return [name for name in columns if name not in {"Email No.", "Prediction"}]


@st.cache_data
def load_dataset_summary() -> dict[str, float]:
    labels = pd.read_csv(DATA_PATH, usecols=["Prediction"])["Prediction"]
    return {
        "emails": int(len(labels)),
        "spam": int(labels.sum()),
        "legitimate": int((labels == 0).sum()),
        "spam_rate": float(labels.mean()),
    }


def vectorize_text(text: str, feature_names: list[str]) -> pd.DataFrame:
    """Convert raw email text to the word-count representation used in training."""
    counts = Counter(TOKEN_RE.findall(text.lower()))
    row = {feature: counts.get(feature.lower(), 0) for feature in feature_names}
    return pd.DataFrame([row], columns=feature_names, dtype=float)


def matched_contributions(
    vector: pd.DataFrame, model, limit: int = 8
) -> pd.DataFrame:
    coefficients = np.asarray(model.coef_)[0]
    counts = vector.iloc[0].to_numpy()
    matched = counts > 0
    if not matched.any():
        return pd.DataFrame(columns=["Word", "Count", "Effect", "Direction"])

    effects = counts[matched] * coefficients[matched]
    result = pd.DataFrame(
        {
            "Word": vector.columns.to_numpy()[matched],
            "Count": counts[matched].astype(int),
            "Effect": effects,
        }
    )
    result["Direction"] = np.where(
        result["Effect"] >= 0, "Toward spam", "Toward legitimate"
    )
    return result.reindex(result["Effect"].abs().sort_values(ascending=False).index).head(limit)


def predict_spam_probability(vector: pd.DataFrame, model) -> float:
    """Evaluate binary logistic regression from its learned parameters.

    Reading coef_ and intercept_ directly also keeps this small demo compatible
    with model artifacts saved by nearby scikit-learn versions.
    """
    logit = float(vector.iloc[0].to_numpy() @ np.asarray(model.coef_)[0])
    logit += float(np.asarray(model.intercept_)[0])
    # Numerically stable sigmoid.
    if logit >= 0:
        return 1.0 / (1.0 + np.exp(-logit))
    exp_logit = np.exp(logit)
    return float(exp_logit / (1.0 + exp_logit))


st.markdown(
    """
    <style>
    .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px;}
    [data-testid="stSidebar"] {
        background: #e9eef5;
        color: #172033;
        border-right: 1px solid #cbd5e1;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #172033;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: #475569;
    }
    [data-testid="stSidebar"] hr {border-color: #cbd5e1;}
    .hero {
        padding: 2.2rem 2.4rem; border-radius: 22px; color: white;
        background: linear-gradient(120deg, #172554 0%, #3730a3 58%, #0f766e 100%);
        box-shadow: 0 14px 38px rgba(30, 41, 59, .16); margin-bottom: 1.5rem;
    }
    .hero h1 {margin: 0; font-size: 2.55rem; letter-spacing: -.04em;}
    .hero p {margin: .7rem 0 0; color: #dbeafe; font-size: 1.05rem; max-width: 720px;}
    .result-card {padding: 1.5rem; border-radius: 18px; margin: .6rem 0 1rem;}
    .spam {background: #fff1f2; border: 1px solid #fecdd3; color: #9f1239;}
    .safe {background: #ecfdf5; border: 1px solid #a7f3d0; color: #065f46;}
    .result-card h2 {margin: 0 0 .35rem;}
    .result-card p {margin: 0; opacity: .9;}
    .small-note {color: #64748b; font-size: .86rem;}
    div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        padding: 1rem;
        border-radius: 14px;
        color: #172033;
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #172033;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    model = load_model()
    features = load_feature_names()
except Exception as exc:
    st.error(f"Could not start the app: {exc}")
    st.stop()

with st.sidebar:
    st.markdown("## 🛡️ Inbox Shield")
    st.caption("Logistic-regression spam classifier")
    st.divider()
    threshold = st.slider(
        "Spam threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Lower values catch more potential spam; higher values reduce false alarms.",
    )
    st.markdown("#### How to use it")
    st.markdown("1. Paste an email\n2. Select **Analyze email**\n3. Review the score and word signals")
    st.divider()
    st.caption(
        "Demo only · This model recognizes words in its fixed training vocabulary and "
        "should not be used as a production email filter."
    )

st.markdown(
    """
    <section class="hero">
      <div style="font-size:.9rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#99f6e4;">Machine-learning demo</div>
      <h1>See suspicious email before you click.</h1>
      <p>Paste a message to estimate whether it looks like spam, then inspect which recognized words influenced the model.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

analyze_tab, insights_tab, about_tab = st.tabs(
    ["✉️ Analyze", "📊 Model insights", "ℹ️ How it works"]
)

with analyze_tab:
    left, right = st.columns([1.7, 1], gap="large")
    with left:
        example = st.selectbox("Try a sample message", EXAMPLES.keys())
        if "email_text" not in st.session_state:
            st.session_state.email_text = ""
        if example != "Choose an example…" and st.session_state.get("last_example") != example:
            st.session_state.email_text = EXAMPLES[example]
            st.session_state.last_example = example

        email_text = st.text_area(
            "Email content",
            key="email_text",
            height=245,
            placeholder="Paste the subject and body of an email here…",
        )
        submitted = st.button("Analyze email", type="primary", use_container_width=True)

    with right:
        st.markdown("#### What the score means")
        st.markdown(
            "The model returns a probability from **0%** (more legitimate) to "
            "**100%** (more spam-like). The decision changes at the threshold in the sidebar."
        )
        st.info(
            "A prediction is a signal, not proof. Check the sender, destination links, "
            "attachments, and unusual requests before taking action.",
            icon="💡",
        )

    if submitted:
        if not email_text.strip():
            st.warning("Enter some email text before running the analysis.")
        else:
            vector = vectorize_text(email_text, features)
            matched_count = int((vector.iloc[0] > 0).sum())
            spam_probability = predict_spam_probability(vector, model)
            is_spam = spam_probability >= threshold

            label = "Likely spam" if is_spam else "Likely legitimate"
            detail = (
                "This message crossed your selected spam threshold."
                if is_spam
                else "This message stayed below your selected spam threshold."
            )
            st.markdown(
                f'<div class="result-card {"spam" if is_spam else "safe"}">'
                f'<h2>{"⚠️" if is_spam else "✅"} {label}</h2><p>{detail}</p></div>',
                unsafe_allow_html=True,
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("Spam probability", f"{spam_probability:.1%}")
            m2.metric("Decision threshold", f"{threshold:.0%}")
            m3.metric("Vocabulary matches", f"{matched_count:,}")
            st.progress(spam_probability, text="Spam likelihood")

            contributions = matched_contributions(vector, model)
            st.markdown("#### Strongest recognized word signals")
            if contributions.empty:
                st.warning(
                    "None of the words in this message occur in the model's training "
                    "vocabulary, so this prediction mainly reflects the model's baseline."
                )
            else:
                display = contributions.copy()
                display["Effect"] = display["Effect"].map(lambda value: f"{value:+.3f}")
                st.dataframe(
                    display,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Word": st.column_config.TextColumn("Recognized word"),
                        "Effect": st.column_config.TextColumn("Model effect"),
                    },
                )
                st.caption(
                    "Positive effects move the estimate toward spam; negative effects move it toward legitimate. "
                    "Effects combine word frequency with the model's learned coefficient."
                )

with insights_tab:
    summary = load_dataset_summary()
    st.markdown("### Training data at a glance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Emails", f"{summary['emails']:,}")
    c2.metric("Word features", f"{len(features):,}")
    c3.metric("Spam emails", f"{summary['spam']:,}")
    c4.metric("Spam share", f"{summary['spam_rate']:.1%}")

    coefficients = np.asarray(model.coef_)[0]
    influence = pd.DataFrame({"Word": features, "Coefficient": coefficients})
    spam_words = influence.nlargest(12, "Coefficient").sort_values("Coefficient")
    legitimate_words = influence.nsmallest(12, "Coefficient").sort_values("Coefficient", ascending=False)
    chart_left, chart_right = st.columns(2, gap="large")
    with chart_left:
        st.markdown("#### Strongest spam indicators")
        st.bar_chart(spam_words.set_index("Word"), horizontal=True, color="#e11d48")
    with chart_right:
        st.markdown("#### Strongest legitimate indicators")
        st.bar_chart(legitimate_words.set_index("Word"), horizontal=True, color="#059669")
    st.caption(
        "These coefficients describe associations learned from this dataset; they do not "
        "mean a word is inherently spam or safe."
    )

with about_tab:
    st.markdown("### From message to prediction")
    st.markdown(
        """
        1. The message is lowercased and split into words.
        2. Occurrences of the model's 3,000 known words are counted.
        3. Logistic regression combines those counts with learned coefficients.
        4. The resulting probability is compared with your selected threshold.

        **Important limitation:** the original dataset contains precomputed word counts, not
        the original emails. This demo reconstructs compatible counts from pasted text, but
        its tokenization may differ from the process used to create the dataset. Treat results
        as an educational showcase rather than a security decision.
        """
    )
