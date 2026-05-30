import os
import logging
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ACCESS_TOKEN = os.environ.get("ACCESS_TOKEN")
PHONE_NUMBER_ID = os.environ.get("PHONE_NUMBER_ID")
VERIFY_TOKEN = os.environ.get("VERIFY_TOKEN")

WHATSAPP_API_BASE = "https://graph.facebook.com/v19.0"

# Supported media message types grouped by category
AUDIO_TYPES = {"audio", "voice"}
IMAGE_TYPES = {"image"}
VIDEO_TYPES = {"video"}
ALL_MEDIA_TYPES = AUDIO_TYPES | IMAGE_TYPES | VIDEO_TYPES

# Track users who have already received the greeting (in-memory; resets on restart)
greeted_users: set[str] = set()


# ---------------------------------------------------------------------------
# WhatsApp helpers
# ---------------------------------------------------------------------------

def _auth_headers() -> dict:
    return {"Authorization": f"Bearer {ACCESS_TOKEN}"}


def _json_headers() -> dict:
    return {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }


def send_whatsapp_message(to: str, body: str) -> dict:
    """Send a plain-text WhatsApp message to a recipient."""
    url = f"{WHATSAPP_API_BASE}/{PHONE_NUMBER_ID}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": body},
    }
    response = requests.post(url, headers=_json_headers(), json=payload, timeout=15)
    response.raise_for_status()
    logger.info("Message sent to %s", to)
    return response.json()


def download_media(media_id: str) -> bytes:
    """
    Download any media file from the WhatsApp Cloud API.

    Steps:
    1. Resolve the temporary media URL using the media ID.
    2. Stream-download the binary content with the Bearer token.
    """
    headers = _auth_headers()

    meta_resp = requests.get(
        f"{WHATSAPP_API_BASE}/{media_id}",
        headers=headers,
        timeout=15,
    )
    meta_resp.raise_for_status()
    media_url = meta_resp.json().get("url")

    if not media_url:
        raise ValueError(f"Could not resolve media URL for media_id={media_id}")

    media_resp = requests.get(media_url, headers=headers, timeout=60)
    media_resp.raise_for_status()
    logger.info("Downloaded %d bytes for media_id=%s", len(media_resp.content), media_id)
    return media_resp.content


# ---------------------------------------------------------------------------
# Deepfake analysis placeholders
# ---------------------------------------------------------------------------

def analyze_audio(audio_bytes: bytes) -> dict:
    """
    PLACEHOLDER – detect AI-generated voice from audio bytes.

    Replace this body with your chosen provider, e.g.:

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  UncovAI  (https://uncovai.com)                                         │
    │  DEEPFAKE_API_KEY = os.environ.get("DEEPFAKE_API_KEY")                  │
    │  resp = requests.post(                                                   │
    │      "https://api.uncovai.com/v1/detect",                               │
    │      headers={"Authorization": f"Bearer {DEEPFAKE_API_KEY}"},           │
    │      files={"file": ("audio.ogg", audio_bytes, "audio/ogg")},           │
    │      timeout=60,                                                         │
    │  )                                                                       │
    │  data = resp.json()                                                      │
    │  return {                                                                 │
    │      "trust_score": 1.0 - data["deepfake_probability"],                 │
    │      "is_deepfake": data["is_deepfake"],                                │
    │      "confidence": data["confidence"],                                   │
    │  }                                                                       │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  Reality Defender  (https://realitydefender.com)                        │
    │  resp = requests.post(                                                   │
    │      "https://api.realitydefender.com/v1/audio/analyze",               │
    │      headers={"Authorization": f"Bearer {DEEPFAKE_API_KEY}"},           │
    │      files={"audio": audio_bytes},                                       │
    │      timeout=60,                                                         │
    │  )                                                                       │
    │  data = resp.json()                                                      │
    │  trust = data["authenticity_score"]                                      │
    │  return {"trust_score": trust, "is_deepfake": trust < 0.5,              │
    │          "confidence": data["confidence"]}                               │
    └─────────────────────────────────────────────────────────────────────────┘

    Returns:
        dict: trust_score (0.0–1.0), is_deepfake (bool), confidence (0.0–1.0)
    """
    logger.info("analyze_audio called with %d bytes (placeholder)", len(audio_bytes))
    return {"trust_score": 0.18, "is_deepfake": True, "confidence": 0.91}


def analyze_image(image_bytes: bytes) -> dict:
    """
    PLACEHOLDER – detect AI-generated or manipulated images.

    Replace this body with your chosen provider, e.g.:

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  Hive Moderation  (https://hivemoderation.com)                          │
    │  DEEPFAKE_API_KEY = os.environ.get("DEEPFAKE_API_KEY")                  │
    │  resp = requests.post(                                                   │
    │      "https://api.thehive.ai/api/v2/task/sync",                         │
    │      headers={"Authorization": f"Token {DEEPFAKE_API_KEY}"},            │
    │      files={"image": image_bytes},                                       │
    │      timeout=60,                                                         │
    │  )                                                                       │
    │  score = resp.json()["status"][0]["response"]["output"][0]              │
    │           ["classes"][0]["score"]  # ai_generated probability            │
    │  return {"trust_score": 1.0 - score, "is_deepfake": score > 0.5,       │
    │          "confidence": score}                                            │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  Reality Defender  (https://realitydefender.com)                        │
    │  resp = requests.post(                                                   │
    │      "https://api.realitydefender.com/v1/image/analyze",               │
    │      headers={"Authorization": f"Bearer {DEEPFAKE_API_KEY}"},           │
    │      files={"image": image_bytes},                                       │
    │      timeout=60,                                                         │
    │  )                                                                       │
    │  data = resp.json()                                                      │
    │  trust = data["authenticity_score"]                                      │
    │  return {"trust_score": trust, "is_deepfake": trust < 0.5,              │
    │          "confidence": data["confidence"]}                               │
    └─────────────────────────────────────────────────────────────────────────┘

    Returns:
        dict: trust_score (0.0–1.0), is_deepfake (bool), confidence (0.0–1.0)
    """
    logger.info("analyze_image called with %d bytes (placeholder)", len(image_bytes))
    return {"trust_score": 0.22, "is_deepfake": True, "confidence": 0.87}


def analyze_video(video_bytes: bytes) -> dict:
    """
    PLACEHOLDER – detect AI-generated or face-swapped video.

    Replace this body with your chosen provider, e.g.:

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  Reality Defender  (https://realitydefender.com)                        │
    │  DEEPFAKE_API_KEY = os.environ.get("DEEPFAKE_API_KEY")                  │
    │  resp = requests.post(                                                   │
    │      "https://api.realitydefender.com/v1/video/analyze",               │
    │      headers={"Authorization": f"Bearer {DEEPFAKE_API_KEY}"},           │
    │      files={"video": video_bytes},                                       │
    │      timeout=120,                                                        │
    │  )                                                                       │
    │  data = resp.json()                                                      │
    │  trust = data["authenticity_score"]                                      │
    │  return {"trust_score": trust, "is_deepfake": trust < 0.5,              │
    │          "confidence": data["confidence"]}                               │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  Sensity AI  (https://sensity.ai)                                       │
    │  resp = requests.post(                                                   │
    │      "https://api.sensity.ai/v1/detect/video",                          │
    │      headers={"Authorization": f"Bearer {DEEPFAKE_API_KEY}"},           │
    │      files={"video": video_bytes},                                       │
    │      timeout=120,                                                        │
    │  )                                                                       │
    │  data = resp.json()                                                      │
    │  return {"trust_score": 1.0 - data["score"], "is_deepfake": data["fake"]│
    │          "confidence": data["confidence"]}                               │
    └─────────────────────────────────────────────────────────────────────────┘

    Returns:
        dict: trust_score (0.0–1.0), is_deepfake (bool), confidence (0.0–1.0)
    """
    logger.info("analyze_video called with %d bytes (placeholder)", len(video_bytes))
    return {"trust_score": 0.14, "is_deepfake": True, "confidence": 0.94}


# ---------------------------------------------------------------------------
# Reply formatting
# ---------------------------------------------------------------------------

MEDIA_LABELS = {
    "audio": "voice note",
    "voice": "voice note",
    "image": "image",
    "video": "video",
}


def build_trust_reply(result: dict, media_type: str) -> str:
    """Format a user-friendly Trust Score reply tailored to the media type."""
    trust_score: float = result["trust_score"]
    is_deepfake: bool = result["is_deepfake"]
    confidence: float = result["confidence"]

    score_pct = int(trust_score * 100)
    conf_pct = int(confidence * 100)
    label = MEDIA_LABELS.get(media_type, "media file")

    if is_deepfake or trust_score < 0.40:
        return (
            f"🚨 *DEEPFAKE DETECTED – HIGH RISK*\n\n"
            f"🔴 Trust Score: {score_pct}%\n"
            f"📊 Detection Confidence: {conf_pct}%\n\n"
            f"This {label} shows strong signs of being *AI-generated or manipulated*.\n\n"
            f"⚠️ *Do NOT* act on instructions or requests made in this {label}.\n"
            f"This is likely a *scam attempt*. Please report it and stay safe.\n\n"
            "_— TrustGuard SA_"
        )
    elif trust_score < 0.65:
        return (
            f"⚠️ *SUSPICIOUS {label.upper()} – PROCEED WITH CAUTION*\n\n"
            f"🟡 Trust Score: {score_pct}%\n"
            f"📊 Detection Confidence: {conf_pct}%\n\n"
            f"This {label} has characteristics that *may indicate AI manipulation*.\n\n"
            f"Please verify the sender through a separate, trusted channel before acting.\n\n"
            "_— TrustGuard SA_"
        )
    else:
        return (
            f"✅ *{label.upper()} APPEARS AUTHENTIC*\n\n"
            f"🟢 Trust Score: {score_pct}%\n"
            f"📊 Detection Confidence: {conf_pct}%\n\n"
            f"This {label} does *not* appear to be AI-generated or manipulated.\n\n"
            "As always, stay vigilant about unsolicited requests involving money or personal information.\n\n"
            "_— TrustGuard SA_"
        )


# ---------------------------------------------------------------------------
# Webhook routes
# ---------------------------------------------------------------------------

@app.route("/webhook", methods=["GET"])
def verify_webhook():
    """Handle Meta's webhook verification challenge."""
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        logger.info("Webhook verified successfully.")
        return challenge, 200

    logger.warning("Webhook verification failed – token mismatch or wrong mode.")
    return "Forbidden", 403


@app.route("/webhook", methods=["POST"])
def receive_message():
    """
    Receive and process incoming WhatsApp messages.

    Supported media:
    - voice / audio  → AI voice detection
    - image          → AI image / face-swap detection
    - video          → AI video / deepfake detection
    - text           → prompt user to send media
    """
    data = request.get_json(silent=True) or {}

    try:
        entry = data["entry"][0]
        changes = entry["changes"][0]
        value = changes["value"]

        if "messages" not in value:
            return jsonify({"status": "ok"}), 200

        message = value["messages"][0]
        sender: str = message["from"]
        msg_type: str = message["type"]

        logger.info("Incoming message from %s, type=%s", sender, msg_type)

        # ── Greeting ──────────────────────────────────────────────────────
        if sender not in greeted_users:
            greeted_users.add(sender)
            send_whatsapp_message(
                sender,
                (
                    "👋 Welcome to *TrustGuard SA*.\n\n"
                    "Send or forward any suspicious media to verify it:\n\n"
                    "🎙️ *Voice notes* — detect AI-generated voices\n"
                    "🖼️ *Images* — detect AI-generated or manipulated photos\n"
                    "🎥 *Videos* — detect deepfake or face-swapped footage\n\n"
                    "Forward the suspicious media here and we'll give you a Trust Score."
                ),
            )
            if msg_type not in ALL_MEDIA_TYPES:
                return jsonify({"status": "ok"}), 200

        # ── Audio / voice note ────────────────────────────────────────────
        if msg_type in AUDIO_TYPES:
            send_whatsapp_message(sender, "🎙️ Analysing your voice note… Please wait a moment.")
            media_id = message[msg_type]["id"]
            media_bytes = download_media(media_id)
            result = analyze_audio(media_bytes)
            send_whatsapp_message(sender, build_trust_reply(result, msg_type))

        # ── Image ─────────────────────────────────────────────────────────
        elif msg_type in IMAGE_TYPES:
            send_whatsapp_message(sender, "🖼️ Analysing your image for AI manipulation… Please wait.")
            media_id = message["image"]["id"]
            media_bytes = download_media(media_id)
            result = analyze_image(media_bytes)
            send_whatsapp_message(sender, build_trust_reply(result, msg_type))

        # ── Video ─────────────────────────────────────────────────────────
        elif msg_type in VIDEO_TYPES:
            send_whatsapp_message(sender, "🎥 Analysing your video for deepfakes… This may take a moment.")
            media_id = message["video"]["id"]
            media_bytes = download_media(media_id)
            result = analyze_video(media_bytes)
            send_whatsapp_message(sender, build_trust_reply(result, msg_type))

        # ── Text / unsupported ────────────────────────────────────────────
        elif msg_type == "text":
            send_whatsapp_message(
                sender,
                (
                    "Please *send or forward the suspicious media* you want me to check:\n\n"
                    "🎙️ Voice note\n"
                    "🖼️ Image\n"
                    "🎥 Video"
                ),
            )
        else:
            send_whatsapp_message(
                sender,
                (
                    "I can analyse *voice notes*, *images*, and *videos* for AI manipulation.\n"
                    "Please forward the suspicious media here."
                ),
            )

    except (KeyError, IndexError, TypeError) as exc:
        logger.error("Failed to parse webhook payload: %s | raw=%s", exc, data)

    return jsonify({"status": "ok"}), 200


# ---------------------------------------------------------------------------
# Privacy Policy
# ---------------------------------------------------------------------------

PRIVACY_POLICY_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Privacy Policy – TrustGuard SA</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto;
           padding: 0 20px; color: #222; line-height: 1.7; }
    h1   { color: #1a73e8; }
    h2   { color: #333; margin-top: 32px; }
    p    { margin: 12px 0; }
    footer { margin-top: 48px; font-size: 0.85em; color: #888; }
  </style>
</head>
<body>
  <h1>🛡️ TrustGuard SA — Privacy Policy</h1>
  <p><strong>Effective date: 12 May 2026</strong></p>

  <p>TrustGuard SA ("we", "our", "the service") is a WhatsApp-based tool that
  analyses voice notes, images, and videos to detect AI-generated or deepfake
  content. This Privacy Policy explains what data we collect, how we use it,
  and your rights.</p>

  <h2>1. Information We Collect</h2>
  <p>When you interact with TrustGuard SA via WhatsApp we may process:</p>
  <ul>
    <li>Your WhatsApp phone number (sender ID)</li>
    <li>Media files you send or forward (voice notes, images, videos) —
        used solely for deepfake analysis and then discarded</li>
    <li>Analysis results (trust score, timestamp) for service improvement</li>
  </ul>

  <h2>2. How We Use Your Information</h2>
  <ul>
    <li>To perform deepfake / AI-generation detection on submitted media</li>
    <li>To send you analysis results and safety warnings via WhatsApp</li>
    <li>To improve detection accuracy over time</li>
  </ul>

  <h2>3. Data Sharing</h2>
  <p>We do <strong>not</strong> sell or share your personal information with
  third parties, except:</p>
  <ul>
    <li>Third-party deepfake detection APIs (media bytes only, no personal
        identifiers)</li>
    <li>WhatsApp / Meta, as required to deliver messages through their platform</li>
    <li>When required by law</li>
  </ul>

  <h2>4. Data Retention</h2>
  <p>Media files are processed in memory and are <strong>not stored</strong>
  on our servers after analysis. Phone numbers are held in memory only for
  session greeting purposes and are cleared on server restart.</p>

  <h2>5. Your Rights</h2>
  <p>You may request deletion of any data associated with your phone number
  by contacting us. You can stop using the service at any time by not sending
  further messages.</p>

  <h2>6. Security</h2>
  <p>All communication between your device, WhatsApp, and our servers is
  encrypted in transit. We do not store media or personal data persistently.</p>

  <h2>7. Contact</h2>
  <p>For privacy questions or data requests, please contact us via WhatsApp
  or email at: <strong>privacy@trustguardsa.co.za</strong></p>

  <h2>8. Changes to This Policy</h2>
  <p>We may update this policy from time to time. The effective date at the
  top of this page will reflect the latest revision.</p>

  <footer>© 2026 TrustGuard SA. All rights reserved.</footer>
</body>
</html>"""


@app.route("/privacy", methods=["GET"])
def privacy_policy():
    return PRIVACY_POLICY_HTML, 200, {"Content-Type": "text/html; charset=utf-8"}


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "TrustGuard SA",
        "supported_media": ["audio", "voice", "image", "video"],
    }), 200


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
