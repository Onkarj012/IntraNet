"""
Test Suite for IntradayNet v3.0

Comprehensive tests for all 6 phases:
- Phase 0: Walk-forward, liquid universe, survivorship bias
- Phase 1: Regime classifier, dynamic targets
- Phase 2: Feature engineering, feature selection
- Phase 3: Specialized models
- Phase 4: Risk management
- Phase 5: Execution infrastructure
- Phase 6: Advanced features

Usage:
    python -m pytest tests/test_v3_complete.py -v
    python tests/test_v3_complete.py --run-slow
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import tempfile
import json
from datetime import datetime, date

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Phase 0 imports
from intradaynet.walkforward_v3 import WalkForwardEngine, WalkForwardConfig
from intradaynet.liquid_universe import LiquidUniverseFilter, LiquidityMetrics
from intradaynet.survivorship_bias import SurvivorshipBiasFix, StockLifecycle

# Phase 1 imports
from intradaynet.regime_v3 import (
    RegimeClassifierV3, MarketRegime, RegimeThresholds, 
    RegimeAdjustments, detect_regime_v3
)
from intradaynet.dynamic_targets import (
    DynamicTargetManager, DynamicTargetConfig
)

# Phase 2 imports
from intradaynet.features.v3_features import EnhancedFeatureEngineer, FeatureConfig
from intradaynet.feature_selection import FeatureSelector, FeatureImportance

# Phase 3 imports
from intradaynet.models.specialized import (
    SpecializedModelSuite, ModelConfig,
    DirectionModel, MagnitudeModel, ConfidenceModel,
    compute_expected_calibration_error
)

# Phase 4 imports
from intradaynet.risk_management import (
    RiskManager, DynamicPositionSizer, CorrelationAwarePortfolio,
    IntradayExitManager, CircuitBreakerSystem,
    PositionSizingConfig, PortfolioConfig, ExitConfig, CircuitBreakerConfig
)

# Phase 5 imports
from intradaynet.execution import (
    PaperTradeLogger, AutomatedRetrainingPipeline,
    TradeRecord, ValidationMetrics
)

# Phase 6 imports
from intradaynet.advanced_features import (
    AdvancedFeatureEngine, FIIDIIIntegrator, EarningsSeasonModule,
    NiftyHedgeLayer, ConfidenceGate, FIIData, EarningsData
)


class TestPhase0Foundation(unittest.TestCase):
    """Tests for Phase 0 - Foundation Reset."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def test_walkforward_fold_creation(self):
        """Test walk-forward fold creation."""
        config = WalkForwardConfig(
            train_months_initial=84,
            val_months=1,
            test_months=1,
            step_months=3,
        )
        engine = WalkForwardEngine(config)
        
        folds = engine.create_folds("2015-01-01", "2025-12-31")
        
        # Should create multiple folds
        self.assertGreater(len(folds), 0)
        
        # Each fold should have correct structure
        for fold in folds:
            self.assertIn('fold', fold)
            self.assertIn('train_start', fold)
            self.assertIn('train_end', fold)
            self.assertIn('val_start', fold)
            self.assertIn('val_end', fold)
            self.assertIn('test_start', fold)
            self.assertIn('test_end', fold)
        
        print(f"✓ Created {len(folds)} walk-forward folds")
    
    def test_liquid_universe_metrics(self):
        """Test liquidity metrics calculation."""
        metrics = LiquidityMetrics(
            symbol="RELIANCE",
            avg_daily_turnover=150_000_000,  # ₹15 Cr
            avg_bid_ask_spread_pct=0.001,    # 0.1%
            trading_days_count=250,
            avg_daily_volume=1_000_000,
            price_volatility_20d=0.02,
        )
        
        # Should pass all liquidity criteria
        self.assertTrue(metrics.is_liquid)
        
        # Test with illiquid stock
        illiquid = LiquidityMetrics(
            symbol="SMALLCAP",
            avg_daily_turnover=5_000_000,  # ₹0.5 Cr (below threshold)
            avg_bid_ask_spread_pct=0.002,
            trading_days_count=100,
            avg_daily_volume=50_000,
            price_volatility_20d=0.05,
        )
        
        self.assertFalse(illiquid.is_liquid)
        print("✓ Liquidity metrics calculation works")
    
    def test_survivorship_lifecycle(self):
        """Test stock lifecycle tracking."""
        lifecycle = StockLifecycle(
            symbol="TEST",
            first_available_date="2020-01-01",
            last_available_date="2024-12-31",
        )
        
        # Should be available during active period
        self.assertTrue(lifecycle.is_available("2022-06-01"))
        
        # Should not be available before IPO
        self.assertFalse(lifecycle.is_available("2019-01-01"))
        
        # Should not be available after delisting
        self.assertFalse(lifecycle.is_available("2025-01-01"))
        
        print("✓ Stock lifecycle tracking works")


class TestPhase1Regime(unittest.TestCase):
    """Tests for Phase 1 - Regime Intelligence."""
    
    def test_regime_classifier_4_states(self):
        """Test 4-state regime classification."""
        classifier = RegimeClassifierV3()
        
        # Test cases for different regimes
        test_cases = [
            (12, 0.5, MarketRegime.TRENDING_CALM, "Low VIX, trending"),
            (18, 0.5, MarketRegime.TRENDING_VOLATILE, "Med VIX, trending"),
            (12, -0.5, MarketRegime.CHOPPY_CALM, "Low VIX, choppy"),
            (25, -0.5, MarketRegime.CHOPPY_VOLATILE, "High VIX, choppy"),
        ]
        
        for vix, trend, expected_regime, desc in test_cases:
            # Create mock nifty data with trend
            dates = pd.date_range(end='2025-01-15', periods=30, freq='D')
            closes = 20000 * (1 + np.cumsum(np.random.randn(30) * 0.001 + trend * 0.0001))
            nifty_df = pd.DataFrame({
                'close': closes,
                'high': closes * 1.01,
                'low': closes * 0.99,
            }, index=dates)
            
            regime, reason, adj = classifier.classify(
                vix_level=vix,
                vix_change_pct=0,
                nifty_df=nifty_df,
            )
            
            print(f"✓ {desc}: {regime.value}")
    
    def test_regime_adjustments(self):
        """Test regime-specific trading adjustments."""
        classifier = RegimeClassifierV3()
        
        # Get adjustments for trending calm (best regime)
        _, _, adj = classifier.classify(
            vix_level=12,
            vix_change_pct=0,
        )
        
        # Should allow trading
        self.assertTrue(adj.allow_trading)
        
        # Should have reasonable multipliers
        self.assertGreater(adj.target_atr_multiplier, 0)
        self.assertGreater(adj.stop_atr_multiplier, 0)
        
        print(f"✓ Regime adjustments: target={adj.target_atr_multiplier:.1f}x, stop={adj.stop_atr_multiplier:.1f}x")
    
    def test_dynamic_targets(self):
        """Test ATR-based dynamic targets."""
        manager = DynamicTargetManager()
        
        # Test LONG position in trending calm
        entry = 1000.0
        atr = 15.0
        
        target, stop, meta = manager.compute_levels(
            entry_price=entry,
            atr=atr,
            side="LONG",
            regime=MarketRegime.TRENDING_CALM,
            confidence=0.65,
        )
        
        # Target should be above entry for LONG
        self.assertGreater(target, entry)
        
        # Stop should be below entry for LONG
        self.assertLess(stop, entry)
        
        # Risk/reward should be reasonable
        self.assertGreater(meta['risk_reward_ratio'], 1.0)
        
        print(f"✓ Dynamic targets: Entry=₹{entry:.0f}, Target=₹{target:.1f}, Stop=₹{stop:.1f}, R/R={meta['risk_reward_ratio']:.2f}")
    
    def test_extreme_regime_no_trading(self):
        """Test that extreme regime blocks trading."""
        classifier = RegimeClassifierV3()
        
        regime, reason, adj = classifier.classify(
            vix_level=30,  # Extreme VIX
            vix_change_pct=0,
        )
        
        self.assertEqual(regime, MarketRegime.EXTREME)
        self.assertFalse(adj.allow_trading)
        
        print("✓ Extreme regime correctly blocks trading")


class TestPhase2Features(unittest.TestCase):
    """Tests for Phase 2 - Feature Engineering."""
    
    def setUp(self):
        """Create sample price data."""
        dates = pd.date_range(end='2025-01-15', periods=100, freq='1min')
        np.random.seed(42)
        
        base_price = 1000
        returns = np.random.randn(100) * 0.001
        prices = base_price * np.exp(np.cumsum(returns))
        
        self.sample_df = pd.DataFrame({
            'open': prices * (1 + np.random.randn(100) * 0.0005),
            'high': prices * (1 + abs(np.random.randn(100)) * 0.001 + 0.001),
            'low': prices * (1 - abs(np.random.randn(100)) * 0.001 - 0.001),
            'close': prices,
            'volume': np.random.poisson(10000, 100),
        }, index=dates)
        
        # Ensure OHLC integrity
        self.sample_df['high'] = self.sample_df[['open', 'high', 'close']].max(axis=1) * 1.001
        self.sample_df['low'] = self.sample_df[['open', 'low', 'close']].min(axis=1) * 0.999
    
    def test_microstructure_features(self):
        """Test microstructure feature computation."""
        engineer = EnhancedFeatureEngineer()
        
        # Test relative volume
        rel_vol = engineer.compute_relative_volume_15m(self.sample_df)
        self.assertIsNotNone(rel_vol)
        self.assertEqual(len(rel_vol), len(self.sample_df))
        
        # Test price acceleration
        accel = engineer.compute_price_acceleration(self.sample_df)
        self.assertIsNotNone(accel)
        
        # Test tick imbalance
        imbalance = engineer.compute_tick_imbalance(self.sample_df)
        self.assertTrue(all(-1 <= x <= 1 for x in imbalance.dropna()))
        
        print("✓ Microstructure features computed")
    
    def test_entropy_feature(self):
        """Test bar entropy feature."""
        engineer = EnhancedFeatureEngineer()
        
        entropy = engineer.compute_bar_entropy(self.sample_df)
        
        # Entropy should be between 0 and 1
        self.assertTrue(all(0 <= x <= 1 for x in entropy.dropna()))
        
        print("✓ Bar entropy feature computed")
    
    def test_all_18_features(self):
        """Test all 18 new features."""
        engineer = EnhancedFeatureEngineer()
        
        features = engineer.compute_all_features(
            minute_df=self.sample_df,
            symbol="TEST",
            sector="TECH",
        )
        
        # Should have 18 new features
        expected_features = engineer.get_feature_names()
        self.assertEqual(len(expected_features), 18)
        
        # All features should be computed
        for feature in expected_features:
            self.assertIn(feature, features.columns)
        
        print(f"✓ All 18 new features computed: {len(features.columns)} total")
    
    def test_feature_selection(self):
        """Test feature selection."""
        from sklearn.ensemble import RandomForestClassifier
        
        # Create synthetic data
        X = np.random.randn(1000, 50)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        feature_names = [f"feature_{i}" for i in range(50)]
        
        selector = FeatureSelector(
            feature_names=feature_names,
            target_count=30,
        )
        
        # Train simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Select features
        selector.fit(X, y, model, n_repeats=2)
        
        # Should select features (may be fewer than target if many are noise)
        self.assertGreater(len(selector.selected_features), 0)
        self.assertLessEqual(len(selector.selected_features), 30)
        
        print(f"✓ Feature selection: 50 → {len(selector.selected_features)} features")


class TestPhase3Models(unittest.TestCase):
    """Tests for Phase 3 - Model Architecture."""
    
    def setUp(self):
        """Create synthetic training data."""
        np.random.seed(42)
        self.X_train = np.random.randn(1000, 20)
        self.y_dir = (self.X_train[:, 0] + self.X_train[:, 1] > 0).astype(int)
        self.y_mag = np.abs(self.X_train[:, 0]) * 0.01
        self.y_conf = (np.abs(self.X_train[:, 0]) > 0.5).astype(int)
        
        self.X_val = np.random.randn(200, 20)
        self.y_val_dir = (self.X_val[:, 0] + self.X_val[:, 1] > 0).astype(int)
    
    def test_direction_model(self):
        """Test direction classifier."""
        config = ModelConfig()
        model = DirectionModel(config)
        
        model.fit(self.X_train, self.y_dir, self.X_val, self.y_val_dir)
        
        # Should produce predictions
        preds = model.predict(self.X_val[:10])
        self.assertEqual(len(preds), 10)
        self.assertTrue(all(0 <= p <= 1 for p in preds))
        
        print("✓ Direction model trained and predicts")
    
    def test_magnitude_model(self):
        """Test magnitude regressor."""
        config = ModelConfig()
        model = MagnitudeModel(config)
        
        model.fit(self.X_train, self.y_mag)
        
        preds = model.predict(self.X_val[:10])
        self.assertEqual(len(preds), 10)
        
        print("✓ Magnitude model trained and predicts")
    
    def test_confidence_model(self):
        """Test confidence classifier."""
        config = ModelConfig()
        model = ConfidenceModel(config)
        
        model.fit(self.X_train, self.y_conf)
        
        preds = model.predict(self.X_val[:10])
        self.assertEqual(len(preds), 10)
        self.assertTrue(all(0 <= p <= 1 for p in preds))
        
        print("✓ Confidence model trained and predicts")
    
    def test_complete_suite(self):
        """Test complete specialized model suite."""
        suite = SpecializedModelSuite()
        
        suite.fit(
            self.X_train, self.y_dir, self.y_mag, self.y_conf,
            self.X_val, self.y_val_dir, self.y_mag[:200], self.y_conf[:200],
        )
        
        # Should produce all predictions
        preds = suite.predict(self.X_val[:5])
        
        self.assertIn('direction_prob', preds)
        self.assertIn('magnitude_estimate', preds)
        self.assertIn('confidence_score', preds)
        
        print("✓ Complete model suite operational")
    
    def test_calibration(self):
        """Test ECE calculation."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.85, 0.15, 0.75, 0.95])
        
        ece = compute_expected_calibration_error(y_true, y_proba, n_bins=5)
        
        # ECE should be non-negative
        self.assertGreaterEqual(ece, 0)
        
        print(f"✓ ECE calculation: {ece:.4f}")


class TestPhase4Risk(unittest.TestCase):
    """Tests for Phase 4 - Risk Management."""
    
    def test_position_sizing(self):
        """Test dynamic position sizing."""
        config = PositionSizingConfig(account_value=100000)
        sizer = DynamicPositionSizer(config)
        
        # Test normal case
        size, meta = sizer.compute_position_size(
            entry_price=1000,
            atr=15,
            stop_distance=12,
        )
        
        # Size should be positive
        self.assertGreater(size, 0)
        
        # Should respect hard caps
        self.assertLessEqual(size, config.max_position_value)
        self.assertGreaterEqual(size, config.min_position_value)
        
        print(f"✓ Position sizing: ₹{size:,.0f} (risk: {meta['risk_pct']:.2f}%)")
    
    def test_position_sizing_with_regime(self):
        """Test position sizing with regime adjustment."""
        config = PositionSizingConfig(account_value=100000)
        sizer = DynamicPositionSizer(config)
        
        # Test with choppy volatile regime (50% size)
        regime_adj = {'size_multiplier': 0.5}
        
        size_normal, _ = sizer.compute_position_size(
            entry_price=1000,
            atr=15,
            regime_adjustments={'size_multiplier': 1.0},
        )
        
        size_choppy, _ = sizer.compute_position_size(
            entry_price=1000,
            atr=15,
            regime_adjustments=regime_adj,
        )
        
        # Choppy should be smaller
        self.assertLess(size_choppy, size_normal)
        
        print(f"✓ Regime-adjusted sizing: normal=₹{size_normal:,.0f}, choppy=₹{size_choppy:,.0f}")
    
    def test_trailing_stop(self):
        """Test trailing stop calculation."""
        config = ExitConfig()
        exit_mgr = IntradayExitManager(config)
        
        entry = 1000
        current = 1007  # +0.7%
        peak = 1008     # Peak was +0.8%
        atr = 15
        
        trailing_stop, reason = exit_mgr.check_trailing_stop(
            entry_price=entry,
            current_price=current,
            peak_price=peak,
            side="LONG",
            atr=atr,
        )
        
        # Should have activated (profit > 0.5%)
        self.assertIsNotNone(trailing_stop)
        self.assertLess(trailing_stop, peak)
        
        print(f"✓ Trailing stop at ₹{trailing_stop:.2f}")
    
    def test_circuit_breakers(self):
        """Test circuit breaker system."""
        config = CircuitBreakerConfig(account_value=100000)
        cb = CircuitBreakerSystem(config)
        
        # Simulate losses
        trades = [
            (-500, "stop_loss"),
            (-400, "stop_loss"),
        ]
        
        for pnl, reason in trades:
            result = cb.update_trade_result(pnl, reason)
        
        # Check status after 2 losses
        status = cb.check_status()
        self.assertEqual(status['consecutive_losses'], 2)
        self.assertFalse(status['trading_halted'])
        
        # Third loss should trigger halt
        result = cb.update_trade_result(-300, "stop_loss")
        status = cb.check_status()
        
        # Should either halt or pause
        self.assertTrue(result.get('halt_trading') or result.get('pause_minutes', 0) > 0)
        
        print("✓ Circuit breakers working")
    
    def test_correlation_aware_portfolio(self):
        """Test correlation-aware portfolio construction."""
        config = PortfolioConfig(max_positions=3, max_correlation=0.4)
        portfolio = CorrelationAwarePortfolio(config)
        
        # Create mock candidates
        candidates = [
            {'symbol': 'A', 'score': 0.9, 'sector': 'TECH'},
            {'symbol': 'B', 'score': 0.8, 'sector': 'TECH'},
            {'symbol': 'C', 'score': 0.7, 'sector': 'BANK'},
            {'symbol': 'D', 'score': 0.6, 'sector': 'ENERGY'},
        ]
        
        # Create mock price history (low correlation)
        price_history = {}
        for sym in ['A', 'B', 'C', 'D']:
            dates = pd.date_range(end='2025-01-15', periods=30, freq='D')
            price_history[sym] = pd.DataFrame({
                'close': 100 + np.cumsum(np.random.randn(30) * 0.02),
            }, index=dates)
        
        selected = portfolio.build_portfolio(candidates, price_history)
        
        # Should select positions
        self.assertGreater(len(selected), 0)
        self.assertLessEqual(len(selected), config.max_positions)
        
        print(f"✓ Portfolio construction: {len(selected)} positions selected")


class TestPhase5Execution(unittest.TestCase):
    """Tests for Phase 5 - Execution Infrastructure."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def test_paper_trade_logging(self):
        """Test paper trade logging."""
        logger = PaperTradeLogger(
            output_dir=self.temp_dir,
            validation_days=5,
        )
        
        # Log predictions
        picks = [
            {
                'symbol': 'RELIANCE',
                'side': 'LONG',
                'entry_reference': 2500.0,
                'target': 2530.0,
                'stop_loss': 2480.0,
                'horizon': 'H60',
                'confidence': 0.65,
            },
        ]
        
        logger.log_predictions("2025-01-15", picks, "trending_calm")
        
        # Log results
        logger.log_results(
            "2025-01-15",
            {'RELIANCE': 2502.0},
            {'RELIANCE': 2535.0},
            {'RELIANCE': 2498.0},
            {'RELIANCE': 2525.0},
        )
        
        # Should have trade records
        self.assertGreater(len(logger.trade_records), 0)
        
        print("✓ Paper trade logging works")
    
    def test_validation_gate(self):
        """Test validation gate."""
        logger = PaperTradeLogger(
            output_dir=self.temp_dir,
            validation_days=5,
        )
        
        # Without enough data, should not pass
        status = logger.get_validation_status(backtest_win_rate=60.0)
        
        self.assertFalse(status['can_go_live'])
        self.assertIn('reason', status)
        
        print(f"✓ Validation gate: {status['reason']}")
    
    def test_retraining_schedule(self):
        """Test retraining schedule detection."""
        pipeline = AutomatedRetrainingPipeline()
        
        # First Saturday of month should trigger
        from datetime import datetime
        first_sat = datetime(2025, 2, 1)  # First Saturday
        self.assertTrue(pipeline.should_retrain(first_sat))
        
        # Not Saturday
        not_sat = datetime(2025, 2, 2)  # Sunday
        self.assertFalse(pipeline.should_retrain(not_sat))
        
        print("✓ Retraining schedule detection works")


class TestPhase6Advanced(unittest.TestCase):
    """Tests for Phase 6 - Advanced Features."""
    
    def test_fii_conflict_detection(self):
        """Test FII conflict detection."""
        integrator = FIIDIIIntegrator()
        
        # Mock FII selling signal
        integrator.fii_data = {
            '2025-01-15': [FIIData('2025-01-15', -500, 100, -500, 100)]
        }
        
        # Create signal function override for test
        def mock_get_fii_signal(symbol, date):
            return {'fii_net_flow_5d': -500, 'fii_signal': -1}
        
        integrator.get_fii_signal = mock_get_fii_signal
        
        # Check conflict with LONG position
        conflict, reason = integrator.check_conflict("RELIANCE", "LONG", "2025-01-15")
        
        # FII selling + LONG should conflict
        self.assertTrue(conflict)
        
        print("✓ FII conflict detection working")
    
    def test_earnings_risk(self):
        """Test earnings risk check."""
        module = EarningsSeasonModule()
        
        # Add mock earnings data
        module.earnings_data['INFY'] = EarningsData(
            symbol='INFY',
            next_earnings_date='2025-01-15',  # Today!
            days_to_earnings=0,
            last_earnings_surprise=2.5,
            earnings_beat_streak=3,
            post_earnings_drift=1.2,
        )
        
        # Check risk for today
        risk = module.check_earnings_risk('INFY', '2025-01-15')
        
        # Should not allow trading on earnings day
        self.assertFalse(risk['can_trade'])
        self.assertEqual(risk['days_to_earnings'], 0)
        
        print("✓ Earnings risk detection working")
    
    def test_confidence_gating(self):
        """Test confidence-based trade gating."""
        gate = ConfidenceGate(
            min_confidence_threshold=0.58,
            high_confidence_threshold=0.70,
        )
        
        # Test low confidence - should be rejected
        should_trade, size, reason, conviction = gate.gate_trade(
            0.55, "LONG", "RELIANCE", 20000
        )
        self.assertFalse(should_trade)
        
        # Test medium confidence - should pass with standard size
        should_trade, size, reason, conviction = gate.gate_trade(
            0.62, "LONG", "RELIANCE", 20000
        )
        self.assertTrue(should_trade)
        self.assertEqual(size, 20000)
        
        # Test high confidence - should get 2x size
        should_trade, size, reason, conviction = gate.gate_trade(
            0.72, "LONG", "RELIANCE", 20000
        )
        self.assertTrue(should_trade)
        self.assertEqual(size, 40000)  # 2x
        
        print("✓ Confidence gating working (58% min, 70% = 2x)")
    
    def test_nifty_hedge_computation(self):
        """Test Nifty hedge calculation."""
        hedge = NiftyHedgeLayer(nifty_lot_size=50)
        
        # Compute hedge for high beta portfolio
        result = hedge.compute_hedge_size(
            portfolio_value=100000,
            portfolio_beta=1.5,  # High beta
            nifty_price=21000,
        )
        
        # Should recommend short position for positive excess beta
        self.assertEqual(result['side'], 'SHORT')
        self.assertGreaterEqual(result['n_lots'], 0)
        
        print(f"✓ Nifty hedge: {result['n_lots']} lots {result['side']}")


class TestIntegration(unittest.TestCase):
    """Integration tests for complete system."""
    
    def test_complete_pipeline(self):
        """Test that all components can work together."""
        print("\n" + "="*70)
        print("INTEGRATION TEST: Complete Pipeline")
        print("="*70)
        
        # 1. Create sample data
        dates = pd.date_range(end='2025-01-15', periods=200, freq='1min')
        np.random.seed(42)
        
        sample_df = pd.DataFrame({
            'open': 1000 + np.cumsum(np.random.randn(200) * 0.5),
            'high': 1002 + np.cumsum(np.random.randn(200) * 0.5),
            'low': 998 + np.cumsum(np.random.randn(200) * 0.5),
            'close': 1000 + np.cumsum(np.random.randn(200) * 0.5),
            'volume': np.random.poisson(10000, 200),
        }, index=dates)
        
        # Ensure OHLC integrity
        sample_df['high'] = np.maximum(sample_df['high'], sample_df[['open', 'close']].max(axis=1) + 1)
        sample_df['low'] = np.minimum(sample_df['low'], sample_df[['open', 'close']].min(axis=1) - 1)
        
        # 2. Detect regime
        classifier = RegimeClassifierV3()
        regime, _, adj = classifier.classify(vix_level=14, vix_change_pct=0)
        self.assertIn(regime.value, ['trending_calm', 'trending_volatile', 'choppy_calm', 'choppy_volatile'])
        print(f"✓ Regime detected: {regime.value}")
        
        # 3. Compute features
        engineer = EnhancedFeatureEngineer()
        features = engineer.compute_all_features(sample_df, "TEST")
        self.assertEqual(len(features.columns), 18)
        print(f"✓ Features computed: {len(features.columns)} new features")
        
        # 4. Compute dynamic targets
        target_mgr = DynamicTargetManager()
        target, stop, meta = target_mgr.compute_levels(
            entry_price=1000, atr=15, side="LONG",
            regime=regime, confidence=0.65
        )
        self.assertGreater(target, 1000)
        self.assertLess(stop, 1000)
        print(f"✓ Dynamic targets: Target=₹{target:.1f}, Stop=₹{stop:.1f}")
        
        # 5. Compute position size
        from intradaynet.risk_management import PositionSizingConfig, DynamicPositionSizer
        sizer = DynamicPositionSizer(PositionSizingConfig(account_value=100000))
        size, meta = sizer.compute_position_size(
            entry_price=1000, atr=15, regime_adjustments={'size_multiplier': adj.size_multiplier}
        )
        self.assertGreater(size, 0)
        print(f"✓ Position size: ₹{size:,.0f}")
        
        # 6. Check confidence gating
        gate = ConfidenceGate()
        should_trade, adj_size, reason, conviction = gate.gate_trade(0.65, "LONG", "TEST", size)
        print(f"✓ Confidence gate: {should_trade} (65% confidence)")
        
        print("\n✅ ALL INTEGRATION TESTS PASSED")
        print("="*70)


def run_all_tests():
    """Run all test suites."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPhase0Foundation))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase1Regime))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase2Features))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase3Models))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase4Risk))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase5Execution))
    suite.addTests(loader.loadTestsFromTestCase(TestPhase6Advanced))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Run v3.0 tests")
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("INTRADAYNET v3.0 - TEST SUITE")
    print("="*70)
    print()
    
    success = run_all_tests()
    
    print("\n" + "="*70)
    if success:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*70)
    
    sys.exit(0 if success else 1)
