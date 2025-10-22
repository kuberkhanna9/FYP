"""
Test script for the simplified cross-validated pipeline
Demonstrates the key improvements requested:
- 5-fold CV for Random Forest
- Enhanced xLSTM with early stopping
- Confidence score calculation
- Comprehensive backtesting metrics
"""

import time
from signals_backend import run_simplified_pipeline

def test_pipeline():
    """Test the simplified pipeline with performance monitoring"""
    
    print("=" * 60)
    print("TESTING SIMPLIFIED CROSS-VALIDATED PIPELINE")
    print("=" * 60)
    
    # Test with a liquid stock
    ticker = "MSFT"
    
    start_time = time.time()
    print(f"\n🚀 Running simplified pipeline for {ticker}...")
    
    try:
        result = run_simplified_pipeline(ticker)
        
        end_time = time.time()
        runtime = end_time - start_time
        
        print(f"\n⏱️  Runtime: {runtime/60:.2f} minutes")
        
        if result.get('success'):
            print(f"\n✅ Pipeline completed successfully!")
            
            # Model Performance
            print(f"\n📊 MODEL PERFORMANCE:")
            print(f"   RF Test Accuracy: {result.get('model_accuracy', 0):.1%}")
            if result.get('model_info', {}).get('xlstm_available'):
                print(f"   xLSTM Available: ✅")
            else:
                print(f"   xLSTM Available: ❌")
            
            # Trading Performance
            strategy_metrics = result.get('metrics', {}).get('strategy', {})
            print(f"\n💰 TRADING PERFORMANCE:")
            print(f"   Strategy Return: {strategy_metrics.get('total_return_pct', 0):.1f}%")
            print(f"   Sharpe Ratio: {strategy_metrics.get('sharpe_ratio', 0):.2f}")
            if 'num_trades' in strategy_metrics:
                print(f"   Number of Trades: {strategy_metrics['num_trades']}")
            if 'buy_sell_ratio' in strategy_metrics:
                print(f"   BUY/SELL Ratio: {strategy_metrics['buy_sell_ratio']}")
            
            # Confidence Analysis
            print(f"\n🎯 CONFIDENCE ANALYSIS:")
            print(f"   Latest Signal: {result.get('signal_name', 'Unknown')}")
            print(f"   Confidence: {result.get('confidence_pct', 0):.1f}%")
            print(f"   Expected Return: {result.get('expected_pct', 0):.2f}%")
            
            # Performance Goals Check
            print(f"\n🎯 PERFORMANCE GOALS CHECK:")
            runtime_ok = runtime < 600  # < 10 minutes
            print(f"   Runtime <10 min: {'✅' if runtime_ok else '❌'} ({runtime/60:.1f} min)")
            
            model_acc = result.get('model_accuracy', 0)
            accuracy_ok = model_acc > 0.6  # >60% accuracy
            print(f"   Model Accuracy >60%: {'✅' if accuracy_ok else '❌'} ({model_acc:.1%})")
            
            trades_ok = strategy_metrics.get('num_trades', 0) > 5
            print(f"   Active Trading: {'✅' if trades_ok else '❌'} ({strategy_metrics.get('num_trades', 0)} trades)")
            
            if runtime_ok and accuracy_ok and trades_ok:
                print(f"\n🎉 ALL PERFORMANCE GOALS MET!")
            else:
                print(f"\n⚠️  Some performance goals not met - check above")
            
        else:
            print(f"\n❌ Pipeline failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"\n💥 Pipeline crashed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()
