"""
Quick test for the improved xLSTM and signal generation
"""
from signals_backend import run_simplified_pipeline

# Test with a shorter run to verify fixes
print("Testing improved pipeline...")
result = run_simplified_pipeline("AAPL")  # Try AAPL instead

if result.get('success'):
    print(f"\n Success!")
    print(f"Strategy Return: {result['metrics']['strategy']['total_return_pct']:.1f}%")
    print(f"RF Accuracy: {result.get('model_accuracy', 0):.1%}")
    print(f"Signal: {result.get('signal_name', 'Unknown')} ({result.get('confidence_pct', 0):.1f}%)")
    
    # Check if both BUY and SELL signals exist
    strategy_metrics = result['metrics']['strategy']
    if 'buy_sell_ratio' in strategy_metrics:
        print(f"Signal Balance: {strategy_metrics['buy_sell_ratio']}")
else:
    print(f" Failed: {result.get('error')}")
