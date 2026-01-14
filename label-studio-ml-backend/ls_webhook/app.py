from flask import Flask, request, jsonify
from converter import convert_ls_to_yolo

app = Flask(__name__)

@app.route("/webhook", methods=["POST"])
def webhook():
    print("🔥 EXTERNAL WEBHOOK HIT 🔥")
    payload = request.json

    # Safety check
    if payload is None:
        return jsonify({"error": "No payload"}), 400

    event = payload.get("action")
    print(f"[Webhook] Event received: {event}")

    # Only act on completed annotations
    if event in ["ANNOTATION_CREATED", "ANNOTATION_UPDATED"]:
        try:
            convert_ls_to_yolo(payload)
            return jsonify({"status": "GT saved"}), 200
        except Exception as e:
            print(e)
            return jsonify({"error": str(e)}), 500

    return jsonify({"status": "ignored"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9091, debug=True)

