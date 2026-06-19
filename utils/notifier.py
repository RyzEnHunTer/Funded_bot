import requests
import json
import threading

class NotificationCenter:
    """Handles independent push notifications to Telegram or Discord."""
    
    @staticmethod
    def notify(config: dict, message: str):
        """Routes the message to the active platform asynchronously."""
        platform = config.get("platform", "NONE").upper()
        
        if platform == "DISCORD":
            url = config.get("discord_webhook_url", "")
            if url:
                # Run in a background thread to prevent blocking the trading loop
                threading.Thread(target=NotificationCenter._send_discord, args=(url, message), daemon=True).start()
                
        elif platform == "TELEGRAM":
            token = config.get("telegram_bot_token", "")
            chat_id = config.get("telegram_chat_id", "")
            if token and chat_id:
                threading.Thread(target=NotificationCenter._send_telegram, args=(token, chat_id, message), daemon=True).start()

    @staticmethod
    def _send_discord(webhook_url: str, message: str):
        try:
            payload = {"content": message}
            headers = {"Content-Type": "application/json"}
            requests.post(webhook_url, data=json.dumps(payload), headers=headers, timeout=5)
        except Exception as e:
            print(f"\n[Notifier] Failed to send Discord message: {e}")

    @staticmethod
    def _send_telegram(token: str, chat_id: str, message: str):
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            payload = {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            requests.post(url, json=payload, timeout=5)
        except Exception as e:
            print(f"\n[Notifier] Failed to send Telegram message: {e}")
