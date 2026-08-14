import os
import re
import numpy as np
import scipy.io
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
def apply_kmeans_feature_reduction(X, n_clusters=8):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X.T)
    clustered = []
    for cid in range(n_clusters):
        mask = kmeans.labels_ == cid
        if mask.sum() > 0:
            clustered.append(X.T[mask].mean(axis=0))
    return np.array(clustered).T

app = Flask(__name__, static_folder='static')
CORS(app)

DATA_DIR = os.path.join(os.path.dirname(__file__), "dataset")
EXTRACTED_1S_DIR = os.path.join(DATA_DIR, "ExtractedFeatures_1s")
EXTRACTED_4S_DIR = os.path.join(DATA_DIR, "ExtractedFeatures_4s")
# Standard 62 channels list in the exact order of the SEED dataset
SEED_CHANNEL_LIST = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4',
    'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3',
    'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ',
    'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
    'CB1', 'O1', 'OZ', 'O2', 'CB2'
]

# 9x9 grid layout for topomap
SEED_LOCATION_LIST = [
    ['-', '-', '-', 'FP1', 'FPZ', 'FP2', '-', '-', '-'],
    ['-', '-', '-', 'AF3', '-', 'AF4', '-', '-', '-'],
    ['F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8'],
    ['FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8'],
    ['T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8'],
    ['TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8'],
    ['P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8'],
    ['-', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', '-'],
    ['-', '-', 'CB1', 'O1', 'OZ', 'O2', 'CB2', '-', '-']
]

# Label mapping
LABEL_NAMES = {1: "Positive", 0: "Neutral", -1: "Negative"}


def get_data_dir(window_size='1s'):
    """Return the correct dataset directory based on window size."""
    if window_size == '4s':
        return EXTRACTED_4S_DIR
    return EXTRACTED_1S_DIR


def scan_dataset(window_size='1s'):
    """Scan the dataset folder to locate subject session files and labels."""
    data_dir = get_data_dir(window_size)

    if not os.path.exists(data_dir):
        return None, f"ExtractedFeatures_{window_size} directory not found at {data_dir}"

    # Load ground truth labels
    label_path = os.path.join(data_dir, "label.mat")
    if not os.path.exists(label_path):
        return None, f"label.mat not found in {data_dir}"

    try:
        label_data = scipy.io.loadmat(label_path)
        labels = label_data["label"][0].tolist()
    except Exception as e:
        return None, f"Error reading label.mat: {str(e)}"

    # Scan for subject .mat files
    subject_files = {}
    pattern = re.compile(r"^(\d+)_(\d+)\.mat$")

    for filename in os.listdir(data_dir):
        match = pattern.match(filename)
        if match:
            sub_id = int(match.group(1))
            date = match.group(2)
            if sub_id not in subject_files:
                subject_files[sub_id] = []
            subject_files[sub_id].append({
                "date": date,
                "filename": filename,
                "filepath": os.path.join(data_dir, filename)
            })

    for sub_id in subject_files:
        subject_files[sub_id] = sorted(subject_files[sub_id], key=lambda x: x["date"])

    return {"labels": labels, "subjects": subject_files}, None


def build_classifier(classifier_name):
    """Return the chosen sklearn classifier."""
    if classifier_name == 'svm_rbf':
        return SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    elif classifier_name == 'random_forest':
        return RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        return SVC(kernel='linear', C=1.0, probability=True, random_state=42)


def extract_band_features(target_de_raw):
    """Extract per-band DE features for topomap visualization."""
    bands_keys = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    features_by_band = {}
    average_de_by_band = {}

    for band_idx, band_name in enumerate(bands_keys):
        band_data = target_de_raw[:, :, band_idx]           # shape: (62, T)
        features_by_band[band_name] = band_data.T.tolist()  # (T, 62)

    average_de = np.mean(target_de_raw, axis=1)             # (62, 5_bands)
    for band_idx, band_name in enumerate(bands_keys):
        average_de_by_band[band_name] = average_de[:, band_idx].tolist()

    return features_by_band, average_de_by_band


def ordered_probs_list(probs, classes):
    """Reorder probabilities as [neg, neutral, pos] = [-1, 0, 1]."""
    result = []
    for row in probs:
        prob_dict = {classes[idx]: val for idx, val in enumerate(row)}
        result.append([
            prob_dict.get(-1, 0.0),
            prob_dict.get(0, 0.0),
            prob_dict.get(1, 0.0)
        ])
    return result


# ─────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────

@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)


@app.route('/api/status', methods=['GET'])
def get_status():
    """Return available subjects/sessions for both 1s and 4s datasets."""
    window_size = request.args.get('window_size', '1s')
    data, err = scan_dataset(window_size)
    if err:
        return jsonify({"success": False, "error": err}), 400

    subjects_list = []
    for sub_id in sorted(data["subjects"].keys()):
        sessions = [
            {"session_id": idx + 1, "date": s["date"]}
            for idx, s in enumerate(data["subjects"][sub_id])
        ]
        subjects_list.append({
            "subject_id": sub_id,
            "name": f"Subject {sub_id}",
            "sessions": sessions
        })

    return jsonify({
        "success": True,
        "window_size": window_size,
        "subjects": subjects_list,
        "labels": data["labels"]
    })


@app.route('/api/predict', methods=['POST'])
def predict_emotion():
    params = request.get_json() or {}
    subject_id    = params.get('subject_id')
    session_idx   = params.get('session_idx')
    trial_idx     = params.get('trial_id')
    feature_type  = params.get('feature_type', 'de_LDS')
    classifier_name = params.get('classifier', 'svm_linear')
    split_mode    = params.get('split_mode', 'leave_one_out')
    window_size   = params.get('window_size', '1s')   # NEW: '1s' or '4s'

    if not all([subject_id, session_idx is not None, trial_idx]):
        return jsonify({"success": False,
                        "error": "Missing parameters: subject_id, session_idx, or trial_id"}), 400

    subject_id  = int(subject_id)
    session_idx = int(session_idx)
    trial_idx   = int(trial_idx)

    # Validate window_size
    if window_size not in ('1s', '4s'):
        return jsonify({"success": False, "error": "window_size must be '1s' or '4s'"}), 400

    data, err = scan_dataset(window_size)
    if err:
        return jsonify({"success": False, "error": err}), 400

    if subject_id not in data["subjects"]:
        return jsonify({"success": False, "error": f"Subject {subject_id} not found"}), 404

    sessions = data["subjects"][subject_id]
    if session_idx < 0 or session_idx >= len(sessions):
        return jsonify({"success": False,
                        "error": f"Session index {session_idx} out of range"}), 404

    selected_session = sessions[session_idx]
    filepath = selected_session["filepath"]
    labels   = data["labels"]

    try:
        mat_data = scipy.io.loadmat(filepath)
    except Exception as e:
        return jsonify({"success": False, "error": f"Error loading .mat file: {str(e)}"}), 500

    # Verify feature type exists
    test_key = f"{feature_type}1"
    if test_key not in mat_data:
        available = [k for k in mat_data.keys() if not k.startswith("__")]
        prefixes  = sorted(set(re.sub(r'\d+$', '', k) for k in available))
        return jsonify({
            "success": False,
            "error": f"Feature '{feature_type}' not found. Available: {prefixes}"
        }), 400

    # ── SPLIT MODE: 10 Train / 5 Test ──────────────────────────────────────
    if split_mode == '10_5_split':
        if trial_idx < 11 or trial_idx > 15:
            trial_idx = 11

        X_train_list, y_train_list = [], []
        for i in range(1, 11):
            feat = mat_data[f"{feature_type}{i}"]
            n_ch, T, n_b = feat.shape
            feat_flat = np.transpose(feat, (1, 0, 2)).reshape(T, -1)
            X_train_list.append(feat_flat)
            y_train_list.append(np.full(T, labels[i - 1]))

        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        n_clusters = int(params.get('n_clusters', 8))
        X_train_scaled = apply_kmeans_feature_reduction(X_train_scaled, n_clusters)

        clf = build_classifier(classifier_name)
        try:
            clf.fit(X_train_scaled, y_train)
        except Exception as e:
            return jsonify({"success": False,
                            "error": f"Failed to train classifier: {str(e)}"}), 500

        test_summary    = []
        correct_count   = 0
        target_de_raw   = None
        y_pred_active   = None
        ordered_probs_active = None

        for j in range(11, 16):
            feat_j = mat_data[f"{feature_type}{j}"]
            n_ch_j, T_j, n_b_j = feat_j.shape
            feat_j_flat = np.transpose(feat_j, (1, 0, 2)).reshape(T_j, -1)

            X_test_scaled_j = scaler.transform(feat_j_flat)
            X_test_scaled_j = apply_kmeans_feature_reduction(X_test_scaled_j, n_clusters)
            pred_j  = clf.predict(X_test_scaled_j)
            probs_j = clf.predict_proba(X_test_scaled_j)

            vals_j, counts_j = np.unique(pred_j, return_counts=True)
            majority_pred_j  = int(vals_j[np.argmax(counts_j)])
            truth_j          = labels[j - 1]
            is_correct_j     = majority_pred_j == truth_j
            if is_correct_j:
                correct_count += 1

            test_summary.append({
                "trial_id":           j,
                "ground_truth_label": truth_j,
                "ground_truth_name":  LABEL_NAMES[truth_j],
                "predicted_label":    majority_pred_j,
                "predicted_name":     LABEL_NAMES[majority_pred_j],
                "is_correct":         bool(is_correct_j),
                "second_accuracy":    float(np.mean(pred_j == truth_j))
            })

            if j == trial_idx:
                target_de_raw        = feat_j
                y_pred_active        = pred_j.tolist()
                ordered_probs_active = ordered_probs_list(
                    probs_j, clf.classes_.tolist())

        overall_acc = correct_count / 5.0
        features_by_band, average_de_by_band = extract_band_features(target_de_raw)

        return jsonify({
            "success":                    True,
            "split_mode":                 "10_5_split",
            "window_size":                window_size,
            "subject_id":                 subject_id,
            "session_idx":                session_idx,
            "session_date":               selected_session["date"],
            "trial_id":                   trial_idx,
            "duration_seconds":           len(y_pred_active),
            "ground_truth_label":         labels[trial_idx - 1],
            "ground_truth_name":          LABEL_NAMES[labels[trial_idx - 1]],
            "predicted_label":            test_summary[trial_idx - 11]["predicted_label"],
            "predicted_name":             test_summary[trial_idx - 11]["predicted_name"],
            "second_by_second_predictions":    y_pred_active,
            "second_by_second_probabilities":  ordered_probs_active,
            "second_by_second_accuracy":  float(np.mean(
                np.array(y_pred_active) == labels[trial_idx - 1])),
            "features_by_band":           features_by_band,
            "average_de_by_band":         average_de_by_band,
            "channels":                   SEED_CHANNEL_LIST,
            "location_grid":              SEED_LOCATION_LIST,
            "overall_test_accuracy":      float(overall_acc),
            "test_trials_summary":        test_summary
        })

    # ── SPLIT MODE: Leave-One-Trial-Out (LOO) ──────────────────────────────
    else:
        X_train_list, y_train_list = [], []
        X_test        = None
        y_test_truth  = labels[trial_idx - 1]
        target_de_raw = None

        for i in range(1, 16):
            feat = mat_data[f"{feature_type}{i}"]
            n_ch, T, n_b = feat.shape
            feat_flat    = np.transpose(feat, (1, 0, 2)).reshape(T, -1)

            if i == trial_idx:
                X_test        = feat_flat
                target_de_raw = feat
            else:
                X_train_list.append(feat_flat)
                y_train_list.append(np.full(T, labels[i - 1]))

        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # K-Means feature clustering
        n_clusters = int(params.get('n_clusters', 8))
        X_train_scaled = apply_kmeans_feature_reduction(X_train_scaled, n_clusters)
        X_test_scaled = apply_kmeans_feature_reduction(X_test_scaled, n_clusters)

        clf = build_classifier(classifier_name)
        try:
            clf.fit(X_train_scaled, y_train)
        except Exception as e:
            return jsonify({"success": False,
                            "error": f"Failed to train classifier: {str(e)}"}), 500

        y_pred  = clf.predict(X_test_scaled).tolist()
        probs   = clf.predict_proba(X_test_scaled)
        classes = clf.classes_.tolist()

        ordered_probs = ordered_probs_list(probs, classes)

        vals, counts  = np.unique(y_pred, return_counts=True)
        majority_pred = int(vals[np.argmax(counts)])
        second_acc    = float(np.mean(np.array(y_pred) == y_test_truth))

        features_by_band, average_de_by_band = extract_band_features(target_de_raw)

        return jsonify({
            "success":                    True,
            "split_mode":                 "leave_one_out",
            "window_size":                window_size,
            "subject_id":                 subject_id,
            "session_idx":                session_idx,
            "session_date":               selected_session["date"],
            "trial_id":                   trial_idx,
            "duration_seconds":           len(y_pred),
            "ground_truth_label":         y_test_truth,
            "ground_truth_name":          LABEL_NAMES[y_test_truth],
            "predicted_label":            majority_pred,
            "predicted_name":             LABEL_NAMES[majority_pred],
            "second_by_second_predictions":    y_pred,
            "second_by_second_probabilities":  ordered_probs,
            "second_by_second_accuracy":  second_acc,
            "features_by_band":           features_by_band,
            "average_de_by_band":         average_de_by_band,
            "channels":                   SEED_CHANNEL_LIST,
            "location_grid":              SEED_LOCATION_LIST
        })


if __name__ == '__main__':
    app.run(debug=True, port=8080)