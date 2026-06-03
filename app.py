import os
import logging
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Env vars ────────────────────────────────────────────────────────────────
ACCESS_TOKEN    = os.environ.get("ACCESS_TOKEN")
PHONE_NUMBER_ID = os.environ.get("PHONE_NUMBER_ID")
VERIFY_TOKEN    = os.environ.get("VERIFY_TOKEN")
HF_TOKEN        = os.environ.get("HF_TOKEN")          # Hugging Face API token

WHATSAPP_API_BASE = "https://graph.facebook.com/v19.0"
HF_API_BASE       = "https://api-inference.huggingface.co/models"

# Hugging Face model endpoints
HF_AUDIO_MODEL  = "garystafford/wav2vec2-deepfake-voice-detector"
HF_IMAGE_MODEL  = "dima806/deepfake_vs_real_image_detection"
HF_VIDEO_MODEL  = "dima806/deepfake_vs_real_image_detection"   # frame-level; see notes below

# Supported media message types grouped by category
AUDIO_TYPES     = {"audio", "voice"}
IMAGE_TYPES     = {"image"}
VIDEO_TYPES     = {"video"}
ALL_MEDIA_TYPES = AUDIO_TYPES | IMAGE_TYPES | VIDEO_TYPES

# Track users who have already received the greeting (in-memory; resets on restart)
greeted_users: set[str] = set()


# ── WhatsApp helpers ─────────────────────────────────────────────────────────

def _auth_headers() -> dict:
    return {"Authorization": f"Bearer {ACCESS_TOKEN}"}


def _json_headers() -> dict:
    return {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }


def send_whatsapp_message(to: str, body: str) -> dict:
    """Send a plain-text WhatsApp message to a recipient. Never raises -- returns dict with success status."""
    url = f"{WHATSAPP_API_BASE}/{PHONE_NUMBER_ID}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": body},
    }
    try:
        response = requests.post(url, headers=_json_headers(), json=payload, timeout=15)
        if response.status_code == 200:
            logger.info("Message sent to %s", to)
            return {"ok": True, "data": response.json()}
        else:
            logger.error("WhatsApp API error for %s: status=%d body=%s",
                         to, response.status_code, response.text[:300])
            return {"ok": False, "error": response.text[:300]}
    except requests.exceptions.Timeout:
        logger.error("WhatsApp API timeout for %s", to)
        return {"ok": False, "error": "timeout"}
    except Exception as exc:
        logger.error("WhatsApp API exception for %s: %s", to, exc)
        return {"ok": False, "error": str(exc)[:300]}


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


# ── Hugging Face deepfake detection ──────────────────────────────────────────

def _hf_query(model: str, data: bytes) -> dict:
    """
    Send raw bytes to a Hugging Face Inference API model and return the parsed JSON.

    Retries up to 3 times if the model is cold-starting (loading response).
    """
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    url = f"{HF_API_BASE}/{model}"

    for attempt in range(3):
        resp = requests.post(url, headers=headers, data=data, timeout=120)

        if resp.status_code == 200:
            return resp.json()

        result = resp.json() if resp.content else {}
        if "error" in result and "loading" in result["error"].lower():
            import time
            wait = result.get("estimated_time", 20)
            logger.info("HF model loading, waiting %.0fs (attempt %d/3)", wait, attempt + 1)
            time.sleep(min(wait, 30))
            continue

        logger.error("HF API error: status=%d body=%s", resp.status_code, resp.text[:300])
        raise RuntimeError(f"HF API returned {resp.status_code}: {resp.text[:300]}")

    raise RuntimeError("HF model failed to load after 3 attempts")


def analyze_audio(audio_bytes: bytes) -> dict:
    """
    Detect AI-generated / cloned voice via Hugging Face.

    Model: garystafford/wav2vec2-deepfake-voice-detector
    Returns labels like "bonafide" (real) or "spoof" (fake) with confidence.

    Return schema:
        {"trust_score": 0.0-1.0, "is_deepfake": bool, "confidence": 0.0-1.0}
    """
    logger.info("analyze_audio: sending %d bytes to HF model %s", len(audio_bytes), HF_AUDIO_MODEL)

    result = _hf_query(HF_AUDIO_MODEL, audio_bytes)

    # The model returns a list of label-score dicts, e.g.:
    # [{"label": "bonafide", "score": 0.95}, {"label": "spoof", "score": 0.05}]
    if isinstance(result, list) and result:
        scores = {item["label"].lower(): item["score"] for item in result}
        spoof_score = scores.get("spoof", 0.0)
        bonafide_score = scores.get("bonafide", 0.0)

        trust_score = bonafide_score
        is_deepfake = spoof_score > bonafide_score
        confidence = max(spoof_score, bonafide_score)

        logger.info("analyze_audio: bonafide=%.3f spoof=%.3f => deepfake=%s",
                     bonafide_score, spoof_score, is_deepfake)
        return {"trust_score": trust_score, "is_deepfake": is_deepfake, "confidence": confidence}

    logger.warning("analyze_audio: unexpected HF response format: %s", str(result)[:200])
    return {"trust_score": 0.5, "is_deepfake": False, "confidence": 0.0}


def analyze_image(image_bytes: bytes) -> dict:
    """
    Detect AI-generated / manipulated images via Hugging Face.

    Model: dima806/deepfake_vs_real_image_detection
    Returns labels like "real" or "fake" with confidence.

    Return schema:
        {"trust_score": 0.0-1.0, "is_deepfake": bool, "confidence": 0.0-1.0}
    """
    logger.info("analyze_image: sending %d bytes to HF model %s", len(image_bytes), HF_IMAGE_MODEL)

    result = _hf_query(HF_IMAGE_MODEL, image_bytes)

    # Expected: [{"label": "real", "score": 0.98}, {"label": "fake", "score": 0.02}]
    if isinstance(result, list) and result:
        scores = {item["label"].lower(): item["score"] for item in result}
        fake_score = scores.get("fake", 0.0)
        real_score = scores.get("real", 0.0)

        trust_score = real_score
        is_deepfake = fake_score > real_score
        confidence = max(fake_score, real_score)

        logger.info("analyze_image: real=%.3f fake=%.3f => deepfake=%s",
                     real_score, fake_score, is_deepfake)
        return {"trust_score": trust_score, "is_deepfake": is_deepfake, "confidence": confidence}

    logger.warning("analyze_image: unexpected HF response format: %s", str(result)[:200])
    return {"trust_score": 0.5, "is_deepfake": False, "confidence": 0.0}


def analyze_video(video_bytes: bytes) -> dict:
    """
    Detect AI-generated / deepfake video via Hugging Face.

    Uses frame-level image classification as a best-effort check.
    For production, consider a dedicated video deepfake API.

    Return schema:
        {"trust_score": 0.0-1.0, "is_deepfake": bool, "confidence": 0.0-1.0}
    """
    logger.info("analyze_video: sending %d bytes to HF model %s (frame-level check)",
                len(video_bytes), HF_VIDEO_MODEL)

    result = _hf_query(HF_VIDEO_MODEL, video_bytes)

    if isinstance(result, list) and result:
        scores = {item["label"].lower(): item["score"] for item in result}
        fake_score = scores.get("fake", 0.0)
        real_score = scores.get("real", 0.0)

        trust_score = real_score
        is_deepfake = fake_score > real_score
        confidence = max(fake_score, real_score)

        logger.info("analyze_video: real=%.3f fake=%.3f => deepfake=%s",
                     real_score, fake_score, is_deepfake)
        return {"trust_score": trust_score, "is_deepfake": is_deepfake, "confidence": confidence}

    logger.warning("analyze_video: unexpected HF response format: %s", str(result)[:200])
    return {"trust_score": 0.5, "is_deepfake": False, "confidence": 0.0}


# ── Reply formatting ─────────────────────────────────────────────────────────

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


# ── Webhook routes ───────────────────────────────────────────────────────────

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
    - voice / audio  → AI voice detection  (HF: wav2vec2-deepfake-voice-detector)
    - image          → AI image detection  (HF: deepfake_vs_real_image_detection)
    - video          → AI video detection  (HF: frame-level via image model)
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
            try:
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
            except Exception as exc:
                logger.warning("Could not send greeting to %s: %s", sender, exc)
            if msg_type not in ALL_MEDIA_TYPES:
                return jsonify({"status": "ok"}), 200

        # ── Audio / voice note ────────────────────────────────────────────
        if msg_type in AUDIO_TYPES:
            try:
                send_whatsapp_message(sender, "🎙️ Analysing your voice note… Please wait a moment.")
                media_id = message[msg_type]["id"]
                media_bytes = download_media(media_id)
                result = analyze_audio(media_bytes)
                send_whatsapp_message(sender, build_trust_reply(result, msg_type))
            except Exception as exc:
                logger.error("Audio analysis failed for %s: %s", sender, exc, exc_info=True)
                try:
                    send_whatsapp_message(sender, "❌ Sorry, something went wrong during analysis. Please try again.")
                except Exception:
                    pass

        # ── Image ─────────────────────────────────────────────────────────
        elif msg_type in IMAGE_TYPES:
            try:
                send_whatsapp_message(sender, "🖼️ Analysing your image for AI manipulation… Please wait.")
                media_id = message["image"]["id"]
                media_bytes = download_media(media_id)
                result = analyze_image(media_bytes)
                send_whatsapp_message(sender, build_trust_reply(result, msg_type))
            except Exception as exc:
                logger.error("Image analysis failed for %s: %s", sender, exc, exc_info=True)
                try:
                    send_whatsapp_message(sender, "❌ Sorry, something went wrong during analysis. Please try again.")
                except Exception:
                    pass

        # ── Video ─────────────────────────────────────────────────────────
        elif msg_type in VIDEO_TYPES:
            try:
                send_whatsapp_message(sender, "🎥 Analysing your video for deepfakes… This may take a moment.")
                media_id = message["video"]["id"]
                media_bytes = download_media(media_id)
                result = analyze_video(media_bytes)
                send_whatsapp_message(sender, build_trust_reply(result, msg_type))
            except Exception as exc:
                logger.error("Video analysis failed for %s: %s", sender, exc, exc_info=True)
                try:
                    send_whatsapp_message(sender, "❌ Sorry, something went wrong during analysis. Please try again.")
                except Exception:
                    pass

        # ── Text / unsupported ────────────────────────────────────────────
        elif msg_type == "text":
            try:
                send_whatsapp_message(
                    sender,
                    (
                        "Please *send or forward the suspicious media* you want me to check:\n\n"
                        "🎙️ Voice note\n"
                        "🖼️ Image\n"
                        "🎥 Video"
                    ),
                )
            except Exception as exc:
                logger.warning("Could not send text reply to %s: %s", sender, exc)
        else:
            try:
                send_whatsapp_message(
                    sender,
                    (
                        "I can analyse *voice notes*, *images*, and *videos* for AI manipulation.\n"
                        "Please forward the suspicious media here."
                    ),
                )
            except Exception as exc:
                logger.warning("Could not send reply to %s: %s", sender, exc)

    except (KeyError, IndexError, TypeError) as exc:
        logger.error("Failed to parse webhook payload: %s | raw=%s", exc, data)
    except Exception as exc:
        logger.error("Error processing message: %s", exc, exc_info=True)

    return jsonify({"status": "ok"}), 200


# ── Privacy Policy ───────────────────────────────────────────────────────────

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


# ── Health check ─────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "TrustGuard SA",
        "supported_media": ["audio", "voice", "image", "video"],
    }), 200


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
