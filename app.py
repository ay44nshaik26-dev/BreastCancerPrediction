import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="MedTech Breast Cancer",
    layout='centered'
)

st.error(
    '⚠️ EDUCATION PROTOTYPE'
    'This tool was built during an Educational workshop facilitated by UOP MedTech. Not been validated for clinical use'
)

st.title(
    "Ayaan's Breast Cancer Predicting Machine"
)

st.caption(
    'Educational demonstration · '
    'Wisconsin Breast Cancer Dataset (UCI, 1993) · '
    'Decision Tree classifier'
)

st.markdown('---')

@st.cache_resource #this ensures that this has to be run only once
def build_model():
  cancer = load_breast_cancer()
  X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
  y = cancer.target   # 0 = malignant, 1 = benign (sklearn's encoding)

  # Same 80/20 split and random seed used in Sessions 1 and 2
  X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=42, stratify=y
  )

  # Train the decision tree model on the training data
  # max_depth=3 keeps the tree shallow to avoid overfitting (as in Session 2)
  clf = DecisionTreeClassifier(max_depth=3, random_state=42)
  clf.fit(X_train, y_train)

  test_accuracy = clf.score(X_test, y_test)
  train_means = X_train.mean()

  return clf, list(cancer.feature_names), train_means, test_accuracy

# Call the function — result is cached after the first load
model, feature_names, feature_means, test_accuracy = build_model() #deciding a function for the basis of the model, to be recalled later on

with st.sidebar:
    st.header("Model Information")
    st.metric(label='Test Accuracy', value=f'{test_accuracy*100:.1f}%') #f string being employed (Logistic Regression (generative ai)). rounding to 1dp and expressing as percentage
    st.metric(label='Training patients', value='455')
    st.metric(label='Test patients', value='114')
    st.metric(label='Total features', value='30 (5 shown)')

    st.markdown('---')
    st.markdown(
        '**What are these measurements?**'
    )
    st.markdown(
        'Values come from digitised images of Fine Needle Aspirate (FNA) '
        'biopsies — a needle-based sampling procedure. '
        'They describe cell size, shape, and texture.'
    )
    st.markdown('---')
    st.warning('⚠️  Not medical advice.')

st.subheader('Cell Measurement Inputs')
st.markdown(
    'Adjust the 5 sliders below. The remaining 25 features from the dataset '
    'are held at their average values from the training data. '
    'The risk indicator updates automatically.'
)
st.markdown(' ')

col1, col2 = st.columns(2)

with col1:
    # Worst radius
    worst_radius = st.slider(
        'Worst Radius (mm)',
        min_value=7.0, max_value=37.0, value=16.0, step=0.1,
        help='Ermmmm.. it is the largest radius measurement from all cells in the biopsy sample'
    )
    # Worst texture
    worst_texture = st.slider(
        'Worst Texture',
        min_value=12.0, max_value=50.0, value=25.0, step=0.5,
        help='Texture irregularity score — basically... higher means rougher, more irregular cells'
    )
    # Worst symmetry
    worst_symmetry = st.slider(
        'Worst Symmetry',
        min_value=0.15, max_value=0.66, value=0.29, step=0.01,
        help='Simon sayes higher value means more asymmetric cells — healthy cells tend to be symmetric'
    )

with col2:
    # Worst concave points
    worst_concave_pts = st.slider(
        'Worst Concave Points',
        min_value=0.00, max_value=0.30, value=0.11, step=0.005,
        help='Number of concave indentations in the most irregular cell boundary (0 = smooth)'
    )
    # Mean concave points
    mean_concave_pts = st.slider(
        'Mean Concave Points',
        min_value=0.00, max_value=0.20, value=0.05, step=0.005,
        help='Dude, it is the average number of concave indentations across all cells in the sample...'
    )

input_row = feature_means.copy()

input_row['worst radius'] = worst_radius
input_row['worst texture'] = worst_texture
input_row['worst symmetry'] = worst_symmetry
input_row['worst concave points'] = worst_concave_pts
input_row['mean concave points'] = mean_concave_pts

# Reshape into a 2D array: sklearn's predict expects shape (n_samples, n_features)
input_array = np.array([input_row[feature_names].values])

probs = model.predict_proba(input_array)[0]
prob_malignant = probs[0]    # probability the pattern matches malignant training examples
prob_benign = probs[1]    # probability the pattern matches benign training examples

# ─────────────────────────Map probability to a plain-language risk band ─────────────────────────

if prob_malignant < 0.25:
    risk_label = '🟢  LOW RISK PATTERN'
    risk_note  = (
        'The entered measurements are more consistent with patterns '
        'seen in benign samples in the training data.'
    )
elif prob_malignant < 0.60:
    risk_label = '🟡  MEDIUM RISK PATTERN'
    risk_note  = (
        'The entered measurements show mixed patterns. '
        'The model is uncertain — values fall between typical benign and malignant ranges.'
    )
else:
    risk_label = '🔴  HIGH RISK PATTERN'
    risk_note  = (
        'The entered measurements are more consistent with patterns '
        'seen in malignant samples in the training data.'
    )

st.markdown('---')

st.subheader('Risk Indicator') #display result
st.markdown(f'## {risk_label}')           # large coloured label
st.progress(float(prob_malignant))         # visual confidence bar (0 = benign, 1 = malignant)
st.markdown(
    f'Malignant pattern score: {prob_malignant:.0%}'
    f' |  '
    f'Benign pattern score: {prob_benign:.0%}'
)
st.info(risk_note)


#Bottom disclaimer, reapeated at the foot of every page
st.markdown('---')
st.error(
    '🚫  This indicator does NOT diagnose cancer. '
    'It is a teaching demonstration only. '
    'Real cancer diagnosis requires clinical examination, imaging, biopsy analysis, '
    'and interpretation by qualified medical professionals. '
    'Please consult a doctor if you have any health concerns.'
)
st.caption(
    'MedTech x Computer Society Workshop | Educational prototype | Not for clinical use | Wisconsin Breast Cancer Dataset'
)
