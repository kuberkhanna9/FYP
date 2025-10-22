#!/usr/bin/env python3
"""
Quick test of enhanced xLSTM performance on AAPL
"""

from signals_backend import run_simplified_pipeline

def test_enhanced_performance():
    print("Testing enhanced xLSTM performance on AAPL...")
    
    result = run_simplified_pipeline('AAPL')
    
    if result.get('success', False):
        print(f"✓ Pipeline successful for {result['ticker']}")
        
        # Check enhanced metrics
        classification_metrics = result['metrics']['classification']
        
        accuracy = classification_metrics['accuracy']
        ensemble_acc = classification_metrics.get('ensemble_accuracy', 'N/A')
        rf_acc = classification_metrics.get('rf_accuracy', 'N/A')
        xlstm_acc = classification_metrics.get('xlstm_accuracy', 'N/A')
        
        print(f'\n--- ENHANCED xLSTM PERFORMANCE ---')
        print(f'Best Model Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)')
        print(f'xLSTM Accuracy: {xlstm_acc}')
        print(f'RF Accuracy: {rf_acc}')
        print(f'Ensemble Accuracy: {ensemble_acc}')
        
        # Signal and confidence
        confidence = result.get('confidence_pct', 0)
        expected = result.get('expected_pct', 0)
        signal_name = result.get('signal_name', 'UNKNOWN')
        
        print(f'\nSignal: {signal_name}')
        print(f'Confidence: {confidence:.1f}%')
        print(f'Expected Return: {expected:.2f}%')
        
        # Strategy performance
        strategy_metrics = result['metrics'].get('strategy', {})
        strategy_return = strategy_metrics.get('total_return_pct', 0)
        sharpe = strategy_metrics.get('sharpe_ratio', 0)
        
        print(f'\nStrategy Return: {strategy_return:.2f}%')
        print(f'Strategy Sharpe: {sharpe:.3f}')
        
        # Check if targets achieved
        target_acc = 0.75
        target_conf = 0.75
        
        xlstm_accuracy = xlstm_acc if isinstance(xlstm_acc, (int, float)) else 0
        confidence_pct = confidence / 100.0
        
        print(f'\n--- TARGET ASSESSMENT ---')
        print(f'Accuracy Target (75%): {"✅" if xlstm_accuracy >= target_acc else "❌"} {xlstm_accuracy*100:.1f}%')
        print(f'Confidence Target (75%): {"✅" if confidence_pct >= target_conf else "❌"} {confidence_pct*100:.1f}%')
        
        if xlstm_accuracy >= target_acc and confidence_pct >= target_conf:
            print('\n🎉 TARGETS ACHIEVED! Enhanced xLSTM is performing as expected.')
        else:
            print(f'\n⚠️  Targets not yet achieved. Continue training or adjust parameters.')
        
        return {
            'accuracy': xlstm_accuracy,
            'confidence': confidence_pct,
            'signal': signal_name,
            'sharpe': sharpe
        }
    else:
        print('✗ Pipeline failed:', result.get('error', 'Unknown error'))
        return None

if __name__ == "__main__":
    test_enhanced_performance()
