import sys
import time
import traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from risk.state_manager import BotState
from data.mt5_loader import initialize_mt5
import MetaTrader5 as mt5
from scripts.live_bot import run_bot
from scripts.dashboard_server import start_dashboard_server

def main():
    print("=" * 60)
    print("  ANTIGRAVITY HEADLESS AUTOPILOT ENGINE")
    print("=" * 60)
    
    # Start the web dashboard in the background
    from scripts.dashboard_server import start_dashboard_server
    _, public_url = start_dashboard_server()
    
    if not initialize_mt5():
        print("❌ Failed to initialize MT5. Ensure terminal is open.")
        sys.exit(1)
        
    acc = mt5.account_info()
    if not acc:
        print("❌ Failed to get account info. Please check MT5 connection.")
        sys.exit(1)
        
    state = BotState(str(acc.login))
    
    # Ensure the user has run the setup wizard at least once
    if state.config.get("starting_balance", 0.0) == 0.0:
        print("⚠️ Bot is not configured! Please run 'python scripts/live_bot.py' interactively first to set up risk parameters.")
        sys.exit(1)
        
    print(f"✅ Connected to MT5 Account: {acc.login}")
    print("🛡️ Global Error Handler Active. Bot will auto-restart on crashes.\n")
    
    if public_url:
        state.log_event(f"🌍 **Live Dashboard Online:** {public_url}")
        
    # Infinite Auto-Restart Loop
    while True:
        try:
            # Refresh balance before starting
            acc = mt5.account_info()
            balance = acc.balance if acc else state.config["starting_balance"]
            
            # Start the main trading engine
            run_bot(state, str(acc.login), balance)
            
        except KeyboardInterrupt:
            print("\n🛑 Shutting down Autopilot gracefully...")
            break
        except Exception as e:
            err_msg = traceback.format_exc()
            print(f"\n🚨 FATAL CRASH ENCOUNTERED:\n{err_msg}")
            
            # Attempt to log the crash to Discord/Telegram
            try:
                state.log_event(f"🚨 BOT CRASHED: {str(e)[:100]}. Auto-restarting in 10 seconds...")
            except:
                pass
                
            print("🔄 Restarting engine in 10 seconds...")
            time.sleep(10)
            
            # Try to reconnect to MT5 if connection was lost
            initialize_mt5()

if __name__ == "__main__":
    main()
