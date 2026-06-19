import json
import os
from datetime import datetime
from typing import Dict, Any, List
from utils.notifier import NotificationCenter

class BotState:
    """
    Persistent State Manager for the MT5 Live Bot.
    Ensures memory of daily trade counts, open positions, and journals
    survives process restarts and crashes.
    """
    def __init__(self, account_id: str):
        self.account_id = account_id
        self.filepath = f"bot_state_{account_id}.json"
        self.last_trade_date = ""
        self.daily_trades_count = 0
        self.managed_positions = {}
        self.trade_journal = []
        self.config = {
            "starting_balance": 0.0,
            "profit_target_pct": 0.0,
            "daily_dd_pct": 4.0,
            "max_dd_pct": 10.0,
            "phase": "CHALLENGE",
            "start_of_day_balance": 0.0,
            "locked_out_for_day": False,
            "use_session_filter": True,
            "notification": {
                "platform": "NONE",
                "discord_webhook_url": "",
                "telegram_bot_token": "",
                "telegram_chat_id": ""
            }
        }
        
        self.log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        self.load()
        self.check_midnight_reset()

    def log_event(self, message: str):
        """Append a clean, human-readable event to today's history log file."""
        current_date = datetime.now().strftime("%Y_%m_%d")
        current_time = datetime.now().strftime("%H:%M:%S")
        log_file = os.path.join(self.log_dir, f"history_{current_date}.txt")
        
        formatted_message = f"[{current_time}] {message}\n"
        
        # Also print to terminal for immediate visibility
        print(formatted_message, end="")
        
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(formatted_message)
        except Exception as e:
            print(f"Error writing to log file: {e}")
            
        # Broadcast via Notification Center
        if "notification" in self.config:
            NotificationCenter.notify(self.config["notification"], message)

    def load(self):
        """Load state from disk."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                    self.last_trade_date = data.get("last_trade_date", "")
                    self.daily_trades_count = data.get("daily_trades_count", 0)
                    self.managed_positions = data.get("managed_positions", {})
                    self.trade_journal = data.get("trade_journal", [])
                    self.config = data.get("config", self.config)
            except Exception as e:
                print(f"Error loading state: {e}. Starting fresh.")
                self.save()
        else:
            self.save()

    def check_phase_upgrade(self, current_balance: float):
        """Checks if the account balance has hit the profit target."""
        if self.config["phase"] == "CHALLENGE" and self.config["starting_balance"] > 0 and self.config["profit_target_pct"] > 0:
            target_amount = self.config["starting_balance"] * (1 + (self.config["profit_target_pct"] / 100))
            if current_balance >= target_amount:
                self.config["phase"] = "FUNDED"
                self.save()
                self.log_event(f"🎉 TARGET REACHED! Account upgraded to FUNDED phase at balance ${current_balance:,.2f}")
                print("\n" + "=" * 60)
                print(" 🏆 CONGRATULATIONS! PROFIT TARGET REACHED! 🏆")
                print("=" * 60 + "\n")

    def save(self):
        """Safely atomic write state to disk."""
        data = {
            "account_id": self.account_id,
            "last_trade_date": self.last_trade_date,
            "daily_trades_count": self.daily_trades_count,
            "managed_positions": self.managed_positions,
            "trade_journal": self.trade_journal,
            "config": self.config
        }
        try:
            temp_path = self.filepath + ".tmp"
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=4)
            os.replace(temp_path, self.filepath)
        except Exception as e:
            print(f"Error saving state: {e}")

    def check_midnight_reset(self, current_balance: float = 0.0) -> bool:
        """Resets the daily trade governor and Kill Switch if it's a new day."""
        current_date = datetime.now().strftime("%Y-%m-%d")
        if self.last_trade_date != current_date:
            self.last_trade_date = current_date
            self.daily_trades_count = 0
            
            # Reset the Floating Kill Switch Floor
            if current_balance > 0:
                self.config["start_of_day_balance"] = current_balance
                self.config["locked_out_for_day"] = False
                
            self.save()
            return True
        return False

    def increment_daily_trades(self):
        self.daily_trades_count += 1
        self.save()

    def add_position(self, ticket: int, data: Dict[str, Any]):
        """Register a new position for MT5 management."""
        self.managed_positions[str(ticket)] = data
        self.save()

    def update_position(self, ticket: int, key: str, value: Any):
        """Update a specific field of an active position (e.g., breakeven_locked)."""
        ticket_str = str(ticket)
        if ticket_str in self.managed_positions:
            self.managed_positions[ticket_str][key] = value
            self.save()

    def remove_position(self, ticket: int):
        """Stop managing a position."""
        ticket_str = str(ticket)
        if ticket_str in self.managed_positions:
            del self.managed_positions[ticket_str]
            self.save()

    def log_trade(self, ticket: int, exit_time: str, reason: str, pnl: float = 0.0):
        """Move a position from active management to the historical journal."""
        ticket_str = str(ticket)
        entry = {
            "ticket": ticket,
            "exit_time": exit_time,
            "reason": reason,
            "pnl": pnl
        }
        # If we had tracking data, merge it into the journal entry
        if ticket_str in self.managed_positions:
            entry.update(self.managed_positions[ticket_str])
            self.remove_position(ticket)
            
        self.trade_journal.append(entry)
        self.save()
