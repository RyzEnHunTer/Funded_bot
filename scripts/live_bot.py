"""
MT5 Live Trading Bot — Fully Automated Execution.

Runs 24/5 in a loop. Every 15 minutes, it fetches data from MT5,
passes it to the ML models, calculates position sizes, and executes
trades automatically on the live/demo account.
"""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings("ignore")

import MetaTrader5 as mt5
from config.settings import (
    PRIMARY_TIMEFRAME, ATR_MULTIPLIER, REWARD_RISK_RATIO, 
    ML_LONG_THRESHOLD, ML_SHORT_THRESHOLD, INITIAL_BALANCE, 
    RISK_PER_TRADE_PCT, MAX_CONCURRENT_POSITIONS, MAX_TRADES_PER_DAY
)
from data.mt5_loader import initialize_mt5, get_mt5_data
from strategy.mt5_executor import execute_trade, close_all_positions, scale_out_position, modify_sl_tp
from ml.features import compute_all_features, get_supertrend_direction
from ml.predictor import MLPredictor
from risk.session_filter import SessionFilter
from risk.state_manager import BotState
from risk.news_filter import NewsFilter
from strategy.position_sizer import calculate_position_size

def get_open_positions_count() -> int:
    positions = mt5.positions_get()
    return len(positions) if positions else 0

def setup_wizard(state, balance):
    print("\n" + "="*50)
    print("   🆕 ACCOUNT SETUP WIZARD")
    print("="*50)
    print(f"Detected Starting Balance: ${balance:,.2f}")
    
    use_balance = input(f"Use this as Starting Balance? (Y/N) [Y]: ").strip().upper()
    if use_balance == 'N':
        try:
            balance = float(input("Enter Starting Balance (e.g. 5000): "))
        except ValueError:
            print("Invalid input, using detected balance.")
            
    print("\n--- Risk Parameters ---")
    try:
        pt = input("Profit Target % (e.g. 14, or 0 if Funded) [14]: ").strip()
        state.config["profit_target_pct"] = float(pt) if pt else 14.0
        
        dd = input("Max Daily Drawdown % (e.g. 4) [4.0]: ").strip()
        state.config["daily_dd_pct"] = float(dd) if dd else 4.0
        
        md = input("Max Total Drawdown % (e.g. 10) [10.0]: ").strip()
        state.config["max_dd_pct"] = float(md) if md else 10.0
    except ValueError:
        print("Invalid input, using defaults.")
        state.config["profit_target_pct"] = 14.0
        state.config["daily_dd_pct"] = 4.0
        state.config["max_dd_pct"] = 10.0
        
    state.config["starting_balance"] = balance
    state.config["phase"] = "CHALLENGE" if state.config["profit_target_pct"] > 0 else "FUNDED"
    
    state.save()
    print("\n✅ Configuration Saved Successfully!")

def notification_setup_wizard(state):
    print("\n" + "="*50)
    print("   📡 NOTIFICATION CENTER SETUP")
    print("="*50)
    print("Current Platform:", state.config.get("notification", {}).get("platform", "NONE"))
    print("\n[1] Discord Webhook")
    print("[2] Telegram Bot")
    print("[3] Disable Notifications")
    
    choice = input("Select platform [1-3]: ").strip()
    
    if choice == '1':
        webhook = input("Enter Discord Webhook URL: ").strip()
        state.config["notification"]["platform"] = "DISCORD"
        state.config["notification"]["discord_webhook_url"] = webhook
        state.save()
        print("\n✅ Discord configured!")
        state.log_event("🔔 Discord Notification Test Successful!")
        
    elif choice == '2':
        token = input("Enter Telegram Bot Token: ").strip()
        chat_id = input("Enter Telegram Chat ID: ").strip()
        state.config["notification"]["platform"] = "TELEGRAM"
        state.config["notification"]["telegram_bot_token"] = token
        state.config["notification"]["telegram_chat_id"] = chat_id
        state.save()
        print("\n✅ Telegram configured!")
        state.log_event("🔔 Telegram Notification Test Successful!")
        
    elif choice == '3':
        state.config["notification"]["platform"] = "NONE"
        state.save()
        print("\n🚫 Notifications Disabled.")

def settings_menu(state, balance):
    while True:
        status_text = "ON" if state.config.get("use_session_filter", True) else "OFF"
        print("\n" + "="*50)
        print("   ⚙️ SETTINGS CONFIGURATION")
        print("="*50)
        print("[1] Edit Account & Risk Parameters")
        print("[2] Edit Notification Settings (Discord/Telegram)")
        print(f"[3] Toggle Session Filter (Current: {status_text})")
        print("[4] Back to Main Menu")
        print("="*50)
        
        choice = input("Select an option [1-4]: ").strip()
        if choice == '1':
            setup_wizard(state, balance)
        elif choice == '2':
            notification_setup_wizard(state)
        elif choice == '3':
            current = state.config.get("use_session_filter", True)
            state.config["use_session_filter"] = not current
            state.save()
            new_status = "ON" if state.config["use_session_filter"] else "OFF"
            print(f"\n✅ Session Filter is now {new_status}!")
        elif choice == '4':
            break
        else:
            print("Invalid choice.")


def main_menu():
    print("\nConnecting to MT5...")
    if not initialize_mt5():
        print("Failed to initialize MT5. Ensure terminal is open and Algo Trading is allowed.")
        return
        
    account_info = mt5.account_info()
    if account_info is None:
        print("Failed to get MT5 account info.")
        mt5.shutdown()
        return
        
    login = str(account_info.login)
    balance = account_info.balance
    
    state = BotState(login)
    
    if state.config["starting_balance"] == 0.0:
        setup_wizard(state, balance)
        
    while True:
        # Refresh balance for menu display
        acc = mt5.account_info()
        if acc: balance = acc.balance
            
        state.check_phase_upgrade(balance)
        
        target_text = f"Target: ${state.config['starting_balance'] * (1 + state.config['profit_target_pct']/100):,.2f}" if state.config['profit_target_pct'] > 0 else "N/A"
        
        print("\n" + "="*50)
        print("  ANTIGRAVITY MT5 TRADING PLATFORM")
        print("="*50)
        print(f"Account : {login}")
        print(f"Balance : ${balance:,.2f}")
        print(f"Phase   : {state.config['phase']} ({target_text})")
        print("-" * 50)
        print("[1] Start Auto-Trading Bot")
        print("[2] View Trade Journal")
        print("[3] Settings Configuration")
        print("[4] Exit")
        print("="*50)
        
        choice = input("Select an option [1-4]: ").strip()
        
        if choice == '1':
            if not account_info.trade_allowed:
                print("\n❌ WARNING: Algo trading is disabled in MT5! Enable it in Tools > Options > Expert Advisors.")
                input("Press Enter to return to menu...")
                continue
                
            if not state.config.get("use_session_filter", True):
                print("\n⚠️ WARNING: Session Filter is currently OFF. The bot will trade 24/5 without session restrictions.")
                confirm = input("Are you sure you want to start? (Y/N): ").strip().upper()
                if confirm != 'Y':
                    continue
                    
            run_bot(state, login, balance)
            break
        elif choice == '2':
            print("\n--- Trade Journal ---")
            if not state.trade_journal:
                print("No trades logged yet.")
            else:
                for t in state.trade_journal[-10:]:
                    print(f"{t.get('exit_time', 'N/A')} | {t.get('pair', 'Unknown')} | {t.get('reason', '')}")
            input("\nPress Enter to continue...")
        elif choice == '3':
            settings_menu(state, balance)
        elif choice == '4':
            print("Exiting platform...")
            mt5.shutdown()
            break
        else:
            print("Invalid choice.")


def run_bot(state, login, balance):
    pairs = ["USDCAD", "EURUSD", "NQ=F"]
    session_filter = SessionFilter(require_overlap_only=False)
    news_filter = NewsFilter()
    
    print("\n" + "=" * 80)
    print(f"  STARTING AUTO-TRADING ENGINE (Account: {login})")
    print("=" * 80)
    print(f"Monitoring Pairs: {', '.join(pairs)}")
    print("Bot is now running. Press Ctrl+C to stop.\n")
    
    try:
        while True:
            now = datetime.now()
            
            # Check Midnight Reset
            current_acc_info = mt5.account_info()
            if current_acc_info:
                if state.check_midnight_reset(current_acc_info.balance):
                    print(f"\n[{now.strftime('%H:%M:%S')}] 🌙 Midnight Reset: Daily limits and Kill Switch floor reset.")
                
            # Check connection every loop
            if mt5.terminal_info() is None:
                print(f"[{now.strftime('%H:%M:%S')}] ❌ ERROR: MT5 CONNECTION LOST! Attempting to reconnect...")
                initialize_mt5()
                time.sleep(5)
                continue
                
            # We want to run this close to the beginning of the 15-minute bar
            # e.g., xx:00, xx:15, xx:30, xx:45
            if now.minute % 15 == 0 and now.second < 15:
                if state.config.get("locked_out_for_day", False):
                    print(f"[{now.strftime('%H:%M:%S')}] 🚨 KILL SWITCH ACTIVE. Bot is locked until midnight.", end="\r")
                    time.sleep(60)
                    continue

                if news_filter.is_news_embargo_active():
                    print(f"[{now.strftime('%H:%M:%S')}] 📰 NEWS EMBARGO ACTIVE. Skipping scan.", end="\r")
                    time.sleep(60)
                    continue

                # Daily Governor Check
                if state.daily_trades_count >= MAX_TRADES_PER_DAY:
                    print(f"[{now.strftime('%H:%M:%S')}] 🛑 Daily Target Reached ({state.daily_trades_count}/{MAX_TRADES_PER_DAY} trades). Sleeping until tomorrow...", end="\r")
                    time.sleep(60)
                    continue
                    
                print(f"\n[{now.strftime('%H:%M:%S')}] 🔄 NEW 15m CANDLE FORMED: Scanning markets...")
                
                # Check global position limit
                open_trades = get_open_positions_count()
                if open_trades >= MAX_CONCURRENT_POSITIONS:
                    print(f"  Max positions reached ({open_trades}/{MAX_CONCURRENT_POSITIONS}). Skipping new trades.")
                    time.sleep(60)
                    continue
                    
                potential_trades = []
                
                for pair in pairs:
                    try:
                        # Fetch last 4000 bars from MT5 (about 60 days of 15m)
                        df = get_mt5_data(pair, "15m", 4000)
                        if df.empty:
                            continue
                            
                        # Compute Features
                        df = compute_all_features(df)
                        current_time = df.index[-1]
                        current_price = df['close'].iloc[-1]
                        
                        # Session Filter (pass symbol for indices support)
                        if state.config.get("use_session_filter", True):
                            if not session_filter.is_tradeable(current_time, symbol=pair):
                                continue # Skip silently if off-session
                            
                        # Predict
                        predictor = MLPredictor(pair, "15m")
                        probas_df = predictor.predict_proba_batch(df.iloc[[-1]])
                        
                        prob_long = probas_df['prob_1'].iloc[0] if 'prob_1' in probas_df else 0.0
                        prob_short = probas_df['prob_-1'].iloc[0] if 'prob_-1' in probas_df else 0.0
                        
                        st_dir = get_supertrend_direction(df).iloc[-1]
                        
                        signal = "NEUTRAL"
                        if prob_long >= ML_LONG_THRESHOLD and st_dir == 1:
                            signal = "LONG"
                        elif prob_short >= ML_SHORT_THRESHOLD and st_dir == -1:
                            signal = "SHORT"
                            
                        
                        if signal == "NEUTRAL":
                            continue
                            
                        # Calculate Trade Parameters
                        atr = df['high'].iloc[-14:] - df['low'].iloc[-14:]
                        atr_val = atr.mean()
                        sl_dist = atr_val * ATR_MULTIPLIER
                        tp_dist = sl_dist * REWARD_RISK_RATIO
                        
                        if signal == "LONG":
                            sl = current_price - sl_dist
                            tp = current_price + tp_dist
                            confidence = prob_long
                        else:
                            sl = current_price + sl_dist
                            tp = current_price - tp_dist
                            confidence = prob_short
                            
                        # Dynamic Risk Configuration
                        daily_dd = state.config.get("daily_dd_pct", 4.0)
                        # We slice the daily drawdown across the max trades per day, keeping a tiny safety buffer
                        dynamic_risk_pct = (daily_dd / 100.0) / MAX_TRADES_PER_DAY
                        
                        # Calculate lot size based on actual account balance
                        lots = calculate_position_size(
                            account_balance=balance,
                            stop_distance_price=sl_dist,
                            pair=pair,
                            risk_pct=dynamic_risk_pct
                        )
                        
                        potential_trades.append({
                            'pair': pair,
                            'signal': signal,
                            'confidence': confidence,
                            'lots': lots,
                            'sl': sl,
                            'tp': tp,
                            'price': current_price
                        })
                        print(f"  >> DETECTED: {signal} {pair} (Confidence: {confidence:.2%})")
                        
                    except Exception as e:
                        print(f"  Error processing {pair}: {e}")
                        
                # ─── Dynamic Portfolio Selection ───
                if potential_trades:
                    # Sort by highest confidence
                    potential_trades.sort(key=lambda x: x['confidence'], reverse=True)
                    best_trade = potential_trades[0]
                    
                    print(f"\n  🏆 BEST SETUP SELECTED: {best_trade['signal']} {best_trade['pair']} (Confidence: {best_trade['confidence']:.2%})")
                    print(f"     Lots: {best_trade['lots']} | SL: {best_trade['sl']:.5f} | TP: {best_trade['tp']:.5f}")
                    
                    # EXECUTE ONLY THE BEST TRADE
                    res = execute_trade(
                        symbol=best_trade['pair'], 
                        signal=best_trade['signal'], 
                        lots=best_trade['lots'], 
                        sl=best_trade['sl'], 
                        tp=best_trade['tp']
                    )
                    if res:
                        state.log_event(f"🟢 OPENED {best_trade['signal']} on {best_trade['pair']} at {res['price']:.5f} (Conf: {best_trade['confidence']:.2%}) | Ticket: {res['ticket']}")
                        state.add_position(res['ticket'], {
                            'breakeven_locked': False,
                            'original_sl_dist': abs(res['price'] - best_trade['sl']),
                            'entry_price': res['price'],
                            'direction': 1 if best_trade['signal'] == "LONG" else -1,
                            'pair': best_trade['pair'],
                            'original_lots': best_trade['lots']
                        })
                        state.increment_daily_trades()
                else:
                    print("  No A+ setups detected on this candle.")
                
                print(f"[{now.strftime('%H:%M:%S')}] ✅ Scan complete. Waiting for next candle...")
                # Sleep for 60 seconds to avoid running multiple times in the same minute
                time.sleep(60)
            else:
                # ─── MT5 Position Manager (Runs Every 5 Seconds) ───
                if now.second % 5 == 0:
                    positions = mt5.positions_get()
                    if positions:
                        active_tickets = []
                        for pos in positions:
                            ticket = pos.ticket
                            ticket_str = str(ticket)
                            active_tickets.append(ticket_str)
                            if ticket_str in state.managed_positions:
                                mgr = state.managed_positions[ticket_str]
                                pair = mgr['pair']
                                tick = mt5.symbol_info_tick(pair)
                                if tick is None:
                                    continue
                                    
                                current_price = tick.bid if mgr['direction'] == 1 else tick.ask
                                
                                # 1. Check 1:1 Scale Out
                                if not mgr['breakeven_locked']:
                                    hit_1_to_1 = False
                                    if mgr['direction'] == 1 and current_price >= mgr['entry_price'] + mgr['original_sl_dist']:
                                        hit_1_to_1 = True
                                    elif mgr['direction'] == -1 and current_price <= mgr['entry_price'] - mgr['original_sl_dist']:
                                        hit_1_to_1 = True
                                        
                                    if hit_1_to_1:
                                        scale_lots = round(mgr['original_lots'] / 2.0, 2)
                                        if scale_lots >= 0.01:
                                            if scale_out_position(ticket, scale_lots):
                                                # Move SL to Breakeven
                                                if modify_sl_tp(ticket, mgr['entry_price'], pos.tp):
                                                    state.update_position(ticket, 'breakeven_locked', True)
                                                    state.log_event(f"🛡️ BREAKEVEN LOCKED & 50% SCALED OUT on {pair}! (Ticket: {ticket})")
                                                    
                                # 2. Check Supertrend Trailing (if breakeven locked)
                                if mgr['breakeven_locked']:
                                    try:
                                        df_st = get_mt5_data(pair, "15m", 100)
                                        if not df_st.empty:
                                            df_st = compute_all_features(df_st)
                                            if '_supertrend_line' in df_st.columns:
                                                st_line = df_st['_supertrend_line'].iloc[-1]
                                                if not pd.isna(st_line):
                                                    if mgr['direction'] == 1 and st_line > pos.sl and current_price > st_line:
                                                        if modify_sl_tp(ticket, st_line, pos.tp):
                                                            state.log_event(f"📈 TRAILING STOP MOVED to {st_line:.5f} for {pair} (Ticket: {ticket})")
                                                    elif mgr['direction'] == -1 and st_line < pos.sl and current_price < st_line:
                                                        if modify_sl_tp(ticket, st_line, pos.tp):
                                                            state.log_event(f"📉 TRAILING STOP MOVED to {st_line:.5f} for {pair} (Ticket: {ticket})")
                                    except Exception as e:
                                        pass # Ignore temporary fetch errors during trailing
                                        
                        # Clean up managed positions that are no longer open
                        # and log them to the trade journal
                        for t in list(state.managed_positions.keys()):
                            if t not in active_tickets:
                                # Position closed! Log it.
                                pair = state.managed_positions[t]['pair']
                                state.log_event(f"🔴 TRADE FINISHED: {pair} closed by MT5 (Ticket: {t}). Logged to permanent journal.")
                                state.log_trade(int(t), now.strftime('%Y-%m-%dT%H:%M:%SZ'), "MT5 Closed")

                # Print a heartbeat every minute so the user knows it's not frozen
                if now.second == 0:
                    current_acc = mt5.account_info()
                    if current_acc:
                        balance = current_acc.balance
                        equity = current_acc.equity
                        state.check_phase_upgrade(balance)
                        
                        # 1. Floating Kill Switch Check
                        start_bal = state.config.get("start_of_day_balance", 0.0)
                        if start_bal > 0 and not state.config.get("locked_out_for_day", False):
                            max_loss = start_bal * (state.config.get("daily_dd_pct", 4.0) / 100.0)
                            death_line = start_bal - max_loss
                            if equity <= death_line:
                                close_all_positions()
                                state.config["locked_out_for_day"] = True
                                state.save()
                                state.log_event(f"🚨 EMERGENCY KILL SWITCH TRIGGERED! Live equity (${equity:,.2f}) dropped below the Death Line (${death_line:,.2f}). All positions closed. Bot locked until tomorrow.")
                                
                        # 2. Friday Flatten Check
                        if now.weekday() == 4 and now.hour >= 23 and now.minute >= 30:
                            if get_open_positions_count() > 0:
                                close_all_positions()
                                state.log_event("🛑 FRIDAY FLATTEN EXECUTED: All positions closed for the weekend (11:30 PM IST).")
                        
                    if state.config.get("locked_out_for_day", False):
                        print(f"[{now.strftime('%H:%M:%S')}] 🚨 KILL SWITCH ACTIVE. Waiting for midnight.", end="\r")
                    elif state.daily_trades_count >= MAX_TRADES_PER_DAY:
                        print(f"[{now.strftime('%H:%M:%S')}] 🛑 Daily Target Reached. Sleeping until midnight...", end="\r")
                    else:
                        mins_left = 15 - (now.minute % 15)
                        print(f"[{now.strftime('%H:%M:%S')}] Phase: {state.config['phase']} | Balance: ${balance:,.2f} | Trades Today: {state.daily_trades_count}/{MAX_TRADES_PER_DAY}. Waiting {mins_left} min...", end="\r")
                # Sleep for 1 second before checking time again
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main_menu()
