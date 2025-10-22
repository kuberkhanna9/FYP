#!/usr/bin/env python3
"""
Quick test to verify metrics alignment after fixes
"""

from signals_backend import run_simplified_pipeline

def test_metrics_alignment():
    print("Testing metrics alignment with AAPL...")
    
    result = run_simplified_pipeline('AAPL')
    
    if result.get('success', False):
        print(f"✓ Pipeline successful for {result['ticker']}")
        
        # Check metrics alignment
        print('\n--- METRICS ALIGNMENT CHECK ---')
        classification_metrics = result['metrics']['classification']
        
        accuracy = classification_metrics['accuracy']
        ensemble_acc = classification_metrics.get('ensemble_accuracy', 'N/A')
        rf_acc = classification_metrics.get('rf_accuracy', 'N/A')
        xlstm_acc = classification_metrics.get('xlstm_accuracy', 'N/A')
        
        print(f'Model Accuracy (displayed): {accuracy:.3f}')
        print(f'Ensemble Accuracy: {ensemble_acc}')
        print(f'RF Accuracy: {rf_acc}')
        print(f'xLSTM Accuracy: {xlstm_acc}')
        
        # Signal info
        confidence = result.get('confidence_pct', 0)
        expected = result.get('expected_pct', 0)
        signal_name = result.get('signal_name', 'UNKNOWN')
        
        print(f'\nSignal: {signal_name}')
        print(f'Confidence: {confidence:.1f}%')
        print(f'Expected Return: {expected:.2f}%')
        
        # Strategy metrics
        strategy_metrics = result['metrics'].get('strategy', {})
        strategy_return = strategy_metrics.get('total_return_pct', 0)
        sharpe = strategy_metrics.get('sharpe_ratio', 0)
        
        print(f'\nStrategy Return: {strategy_return:.2f}%')
        print(f'Strategy Sharpe: {sharpe:.3f}')
        
        print('\n✓ All metrics extracted successfully')
        
        # Return key metrics for comparison
        return {
            'accuracy': accuracy,
            'confidence': confidence,
            'expected': expected,
            'signal': signal_name,
            'sharpe': sharpe,
            'strategy_return': strategy_return
        }
    else:
        print('✗ Pipeline failed:', result.get('error', 'Unknown error'))
        return None

if __name__ == "__main__":
    test_metrics_alignment()
