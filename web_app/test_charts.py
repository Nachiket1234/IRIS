"""Test chart generation to debug empty plots."""

from pathlib import Path
from metrics_analyzer import MetricsAnalyzer
from chart_utils import create_training_curve_plotly, create_variant_comparison_chart

# Initialize
base_dir = Path('..').resolve()
ma = MetricsAnalyzer(base_dir)

print("=" * 80)
print("Testing MetricsAnalyzer")
print("=" * 80)

# Test 1: Get datasets
datasets = ma.get_available_datasets()
print(f"\nAvailable datasets: {datasets}")

if not datasets:
    print("ERROR: No datasets found!")
    exit(1)

test_dataset = datasets[0]
print(f"\nTesting with dataset: {test_dataset}")

# Test 2: Load training metrics
print("\n" + "=" * 80)
print("Test 2: Loading training metrics")
print("=" * 80)

metrics = ma.load_training_metrics(test_dataset)
if metrics:
    print(f"✓ Training metrics loaded")
    print(f"  Keys: {list(metrics.keys())}")
    print(f"  Total iterations: {metrics.get('total_iterations', 'N/A')}")
else:
    print("✗ No training metrics found")

# Test 3: Extract training curves
print("\n" + "=" * 80)
print("Test 3: Extracting training curves")
print("=" * 80)

curves = ma.extract_training_curves(test_dataset)
if curves:
    iterations, losses, dice_scores = curves
    print(f"✓ Training curves extracted")
    print(f"  Iterations: {len(iterations)} points")
    print(f"  Losses: {len(losses)} points")
    print(f"  Dice scores: {len(dice_scores)} points")
    print(f"  Sample iterations: {iterations[:5] if iterations else 'None'}")
    print(f"  Sample losses: {losses[:5] if losses else 'None'}")
    
    # Test chart creation
    if iterations and losses:
        print("\n  Creating training curve plot...")
        try:
            fig = create_training_curve_plotly(iterations, losses, dice_scores, 
                                             title=f"{test_dataset} Training")
            print(f"  ✓ Plot created: {type(fig)}")
            print(f"  Plot data traces: {len(fig.data)}")
        except Exception as e:
            print(f"  ✗ Plot creation failed: {e}")
            import traceback
            traceback.print_exc()
else:
    print("✗ No training curves found")

# Test 4: Load variant comparison
print("\n" + "=" * 80)
print("Test 4: Loading variant comparison")
print("=" * 80)

variant_comp = ma.load_variant_comparison(test_dataset)
if variant_comp:
    print(f"✓ Variant comparison loaded")
    print(f"  Keys: {list(variant_comp.keys())}")
    
    avg_dice = variant_comp.get('average_dice', {})
    print(f"  Average Dice scores:")
    print(f"    One-shot: {avg_dice.get('oneshot', 0):.4f}")
    print(f"    Ensemble: {avg_dice.get('ensemble', 0):.4f}")
    print(f"    Full: {avg_dice.get('full', 0):.4f}")
    
    improvements = variant_comp.get('improvements', {})
    print(f"  Improvements:")
    print(f"    Ensemble vs One-shot: {improvements.get('ensemble_vs_oneshot', 0):.2f}%")
    print(f"    Memory bank: {improvements.get('memory_bank_contribution', 0):.2f}%")
    
    # Test variant chart
    print("\n  Creating variant comparison chart...")
    try:
        oneshot = avg_dice.get('oneshot', 0)
        ensemble = avg_dice.get('ensemble', 0)
        full = avg_dice.get('full', 0)
        
        fig = create_variant_comparison_chart(oneshot, ensemble, full, test_dataset.upper())
        print(f"  ✓ Variant chart created: {type(fig)}")
        print(f"  Chart data traces: {len(fig.data)}")
    except Exception as e:
        print(f"  ✗ Variant chart creation failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("✗ No variant comparison found")

# Test 5: Get summary stats
print("\n" + "=" * 80)
print("Test 5: Getting summary stats")
print("=" * 80)

stats = ma.get_summary_stats(test_dataset)
if stats:
    print(f"✓ Summary stats loaded")
    print(f"  Training available: {stats.get('training_available')}")
    print(f"  Variant comparison available: {stats.get('variant_comparison_available')}")
    print(f"  Best val dice: {stats.get('best_val_dice', 0):.4f}")
else:
    print("✗ No summary stats")

print("\n" + "=" * 80)
print("Tests complete")
print("=" * 80)
