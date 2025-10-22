"""
Test specific scenario: BUY=36.4% vs SELL=31.8% should give BUY for 30-day
"""

import numpy as np

def test_signal_logic():
    print("=== TESTING SPECIFIC SCENARIO ===")
    
    # Simulate
    buy_prob = 0.364  # 36.4%
    sell_prob = 0.318  # 31.8%
    
    print(f"Test probabilities: BUY={buy_prob*100:.1f}%, SELL={sell_prob*100:.1f}%")
    
    # Test each horizon's logic
    horizons = [
        (1, 0.35, 0.45, 0.10, "1-day"),
        (7, 0.32, 0.42, 0.08, "7-day"), 
        (30, 0.28, 0.38, 0.05, "30-day")
    ]
    
    for horizon, buy_thresh, sell_thresh, neutral_thresh, name in horizons:
        print(f"\\n{name} Analysis:")
        print(f"  Thresholds: BUY>{buy_thresh:.2f}, SELL>{sell_thresh:.2f}, neutral_gap>{neutral_thresh:.2f}")
        
        # Test signal generation logic
        buy_exceeds_thresh = buy_prob > buy_thresh
        sell_exceeds_thresh = sell_prob > sell_thresh
        buy_dominates = buy_prob > sell_prob + neutral_thresh
        sell_dominates = sell_prob > buy_prob + neutral_thresh
        
        print(f"  BUY exceeds threshold: {buy_exceeds_thresh} ({buy_prob:.3f} > {buy_thresh:.3f})")
        print(f"  SELL exceeds threshold: {sell_exceeds_thresh} ({sell_prob:.3f} > {sell_thresh:.3f})")
        print(f"  BUY dominates: {buy_dominates} ({buy_prob:.3f} > {sell_prob + neutral_thresh:.3f})")
        print(f"  SELL dominates: {sell_dominates} ({sell_prob:.3f} > {buy_prob + neutral_thresh:.3f})")
        
        # Test improved signal generation logic
        clear_buy = buy_exceeds_thresh and not sell_exceeds_thresh
        clear_sell = sell_exceeds_thresh and not buy_exceeds_thresh
        long_term_buy_pref = (horizon >= 30) and buy_prob > sell_prob
        long_term_sell_pref = (horizon >= 30) and sell_prob > buy_prob
        
        print(f"  Clear BUY case: {clear_buy}")
        print(f"  Clear SELL case: {clear_sell}")
        print(f"  BUY dominates: {buy_dominates}")
        print(f"  SELL dominates: {sell_dominates}")
        if horizon >= 30:
            print(f"  Long-term BUY preference: {long_term_buy_pref}")
            print(f"  Long-term SELL preference: {long_term_sell_pref}")
        
        # Improved signal decision
        if clear_buy:
            signal = 1
            signal_name = "BUY (clear case)"
        elif clear_sell:
            signal = -1
            signal_name = "SELL (clear case)"
        elif buy_dominates:
            signal = 1
            signal_name = "BUY (dominance)"
        elif sell_dominates:
            signal = -1
            signal_name = "SELL (dominance)"
        elif horizon >= 30 and buy_prob > sell_prob:
            signal = 1
            signal_name = "BUY (long-term preference)"
        elif horizon >= 30 and sell_prob > buy_prob:
            signal = -1
            signal_name = "SELL (long-term preference)"
        elif buy_prob > sell_prob and horizon < 30:
            signal = 0
            signal_name = "HOLD (short-term uncertainty)"
        else:
            signal = 0
            signal_name = "HOLD (neutral)"
        
        print(f"  → Final Signal: {signal_name} ({signal})")
        
        # Check if this matches expected behavior
        if name == "30-day" and signal == 1:
            print(f"  ✅ CORRECT: 30-day shows BUY as expected!")
        elif name == "30-day" and signal != 1:
            print(f"  ❌ ISSUE: 30-day should show BUY but shows {signal_name}")

if __name__ == "__main__":
    test_signal_logic()
