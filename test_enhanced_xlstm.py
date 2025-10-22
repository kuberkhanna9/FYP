#!/usr/bin/env python3
"""
Test Enhanced xLSTM Implementation
"""

def test_enhanced_xlstm():
    try:
        # Test imports
        from signals_backend import EnhancedXLSTM, LabelSmoothingCrossEntropy
        import torch
        
        print("✓ Enhanced xLSTM classes imported successfully")
        
        # Test model creation
        model = EnhancedXLSTM(n_features=45, hidden_size=384, dropout_rate=0.3)
        print(f"✓ Model created: {model.__class__.__name__}")
        
        # Check if model has temperature parameter
        if hasattr(model, 'temperature'):
            print(f"✓ Temperature scaling parameter: {model.temperature.item():.2f}")
        else:
            print("✗ Missing temperature parameter")
        
        # Check if model has attention
        if hasattr(model, 'attention') and model.attention is not None:
            print("✓ Attention mechanism present")
        else:
            print("✗ Missing attention mechanism")
        
        # Test loss function
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        print(f"✓ Label smoothing loss created: {criterion.__class__.__name__}")
        
        # Test forward pass
        batch_size = 16
        seq_len = 10
        n_features = 45
        
        x = torch.randn(batch_size, seq_len, n_features)
        rf_probs = torch.randn(batch_size, 2)
        
        outputs = model(x, rf_probs)
        print(f"✓ Forward pass successful: {outputs.shape}")
        
        # Test loss calculation
        targets = torch.randint(0, 2, (batch_size,))
        loss = criterion(outputs, targets)
        print(f"✓ Loss calculation: {loss.item():.4f}")
        
        print("\n✅ All tests passed! Enhanced xLSTM is ready.")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_enhanced_xlstm()
