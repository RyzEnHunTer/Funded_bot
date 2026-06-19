"""
MT5 Executor — Direct execution engine for MetaTrader 5.

Constructs MqlTradeRequest dictionaries and sends orders to the broker.
"""

import MetaTrader5 as mt5

def close_all_positions(symbol: str = None):
    """Close all open positions, optionally filtered by symbol."""
    positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
    
    if positions is None or len(positions) == 0:
        return 0
        
    closed = 0
    for pos in positions:
        tick = mt5.symbol_info_tick(pos.symbol)
        
        type_dict = {
            mt5.ORDER_TYPE_BUY: mt5.ORDER_TYPE_SELL,
            mt5.ORDER_TYPE_SELL: mt5.ORDER_TYPE_BUY
        }
        
        price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
        
        filling_modes = [
            mt5.ORDER_FILLING_FOK,
            mt5.ORDER_FILLING_IOC,
            mt5.ORDER_FILLING_RETURN
        ]
        
        valid_request = None
        for mode in filling_modes:
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": pos.ticket,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": type_dict[pos.type],
                "price": price,
                "deviation": 20,
                "magic": 234000,
                "comment": "ML Python Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mode,
            }
            if mt5.order_check(request) and mt5.order_check(request).retcode == 0:
                valid_request = request
                break
                
        if valid_request is None:
            print(f"Failed to close position {pos.ticket}: Unsupported filling mode")
            continue
        
        result = mt5.order_send(valid_request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            closed += 1
        else:
            print(f"Failed to close position {pos.ticket}: {result.comment}")
            
    return closed

def execute_trade(symbol: str, signal: str, lots: float, sl: float, tp: float, magic_number: int = 123456):
    """
    Send a market order to MT5.
    
    Parameters
    ----------
    symbol : str
        e.g., "EURUSD"
    signal : str
        "LONG" or "SHORT"
    lots : float
        Volume in standard lots
    sl : float
        Stop Loss absolute price
    tp : float
        Take Profit absolute price
    magic_number : int
        Unique identifier for the EA
        
    Returns
    -------
    dict
        Result of the trade execution
    """
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"{symbol} not found, can not call order_check()")
        return None
        
    if not symbol_info.visible:
        print(f"{symbol} is not visible, trying to switch on")
        if not mt5.symbol_select(symbol, True):
            print(f"symbol_select({symbol}) failed, exit")
            return None
            
    point = mt5.symbol_info(symbol).point
    tick = mt5.symbol_info_tick(symbol)
    
    if signal == "LONG":
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
    elif signal == "SHORT":
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
    else:
        print(f"Invalid signal: {signal}")
        return None
        
    # MT5 Brokers support different filling modes. We will try them all to find the supported one.
    filling_modes = [
        mt5.ORDER_FILLING_FOK,
        mt5.ORDER_FILLING_IOC,
        mt5.ORDER_FILLING_RETURN
    ]
    
    valid_request = None
    for mode in filling_modes:
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lots),
            "type": order_type,
            "price": price,
            "sl": float(sl),
            "tp": float(tp),
            "deviation": 20,
            "magic": magic_number,
            "comment": f"ML {signal}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mode,
        }
        
        check = mt5.order_check(request)
        if check and check.retcode == 0:
            valid_request = request
            break
            
    if valid_request is None:
        print(f"Order Check Failed: Unsupported filling mode or invalid parameters.")
        return None
        
    # Send the order
    result = mt5.order_send(valid_request)
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order Send Failed: {result.comment}")
        return None
        
    print(f"Order Placed Successfully: {result.order}")
    return {
        "ticket": result.order,
        "price": result.price,
        "volume": result.volume
    }

def scale_out_position(ticket: int, lots_to_close: float) -> bool:
    """Close a specific volume of an open position."""
    pos = mt5.positions_get(ticket=ticket)
    if not pos or len(pos) == 0:
        print(f"Position {ticket} not found for scale out.")
        return False
        
    pos = pos[0]
    tick = mt5.symbol_info_tick(pos.symbol)
    
    type_dict = {
        mt5.ORDER_TYPE_BUY: mt5.ORDER_TYPE_SELL,
        mt5.ORDER_TYPE_SELL: mt5.ORDER_TYPE_BUY
    }
    
    price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
    
    filling_modes = [
        mt5.ORDER_FILLING_FOK,
        mt5.ORDER_FILLING_IOC,
        mt5.ORDER_FILLING_RETURN
    ]
    
    valid_request = None
    for mode in filling_modes:
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": pos.ticket,
            "symbol": pos.symbol,
            "volume": float(lots_to_close),
            "type": type_dict[pos.type],
            "price": price,
            "deviation": 20,
            "magic": 123456,
            "comment": "Scale Out",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mode,
        }
        if mt5.order_check(request) and mt5.order_check(request).retcode == 0:
            valid_request = request
            break
            
    if valid_request is None:
        print(f"Failed to scale out position {pos.ticket}: Unsupported filling mode")
        return False
    
    result = mt5.order_send(valid_request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"Successfully scaled out {lots_to_close} lots from position {ticket}")
        return True
    else:
        print(f"Failed to scale out position {ticket}: {result.comment}")
        return False

def modify_sl_tp(ticket: int, new_sl: float, new_tp: float) -> bool:
    """Modify the SL and TP of an open position."""
    pos = mt5.positions_get(ticket=ticket)
    if not pos or len(pos) == 0:
        return False
        
    pos = pos[0]
    
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": pos.ticket,
        "symbol": pos.symbol,
        "sl": float(new_sl),
        "tp": float(new_tp),
        "magic": 123456,
    }
    
    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        return True
    else:
        print(f"Failed to modify SL/TP for {ticket}: {result.comment}")
        return False
