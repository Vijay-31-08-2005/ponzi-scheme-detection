from flask import Blueprint, request, jsonify
from train import infer_scheme

api_routes = Blueprint("api_routes", __name__)

@api_routes.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        scheme_text = data.get("scheme_text", "").strip()

        # Validate input
        if not scheme_text:
            return jsonify({"error": "No scheme text provided"}), 400
        if len(scheme_text) < 10:
            return jsonify({"error": "Scheme description is too short. Provide more details."}), 400

        # Use temporary file method (since it worked before)
        temp_file = "temp_scheme.txt"
        with open(temp_file, "w") as f:
            f.write(scheme_text)

        # Run inference
        result = infer_scheme(temp_file, api=True)
        print("[DEBUG]", result)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500
