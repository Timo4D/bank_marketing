"""
Generate visualization for Accuracy Threshold Simulation Results
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load results
df = pd.read_csv('accuracy_threshold_simulation_results.csv')

# Set up the figure
fig, ax1 = plt.subplots(figsize=(12, 7))

# Convert accuracy to percentage for display
accuracy_pct = df['target_accuracy'] * 100

# Colors
color_eff = '#2E86AB'  # Blue for efficiency
color_prob = '#E94F37'  # Red for probability
color_diff = '#1B998B'  # Teal for difference

# Plot sales lines
ax1.plot(accuracy_pct, df['efficiency_sales'], 'o-', color=color_eff, 
         linewidth=2.5, markersize=10, label='Efficiency Strategy Sales', zorder=3)
ax1.axhline(y=df['probability_sales'].iloc[0], color=color_prob, 
            linewidth=2.5, linestyle='--', label=f"Probability Strategy Sales (constant: {df['probability_sales'].iloc[0]})")

# Fill regions
# Below threshold (probability wins) - light red
threshold_mask = df['efficiency_sales'] < df['probability_sales']
ax1.fill_between(accuracy_pct, df['efficiency_sales'], df['probability_sales'].iloc[0],
                 where=threshold_mask, alpha=0.15, color=color_prob, 
                 label='Probability Wins')

# Above threshold (efficiency wins) - light blue  
ax1.fill_between(accuracy_pct, df['efficiency_sales'], df['probability_sales'].iloc[0],
                 where=~threshold_mask, alpha=0.15, color=color_eff,
                 label='Efficiency Wins')

# Mark the threshold
threshold_acc = 75
ax1.axvline(x=threshold_acc, color='#333333', linewidth=2, linestyle=':', alpha=0.7)
ax1.annotate(f'Threshold\n~{threshold_acc}%', xy=(threshold_acc, 80), 
             fontsize=11, ha='center', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))

# Mark current accuracy
current_acc = 57.59
ax1.axvline(x=current_acc, color='#FF6B35', linewidth=2, linestyle='-', alpha=0.8)
ax1.annotate(f'Current\n{current_acc:.1f}%', xy=(current_acc, 55), 
             fontsize=11, ha='center', fontweight='bold', color='#FF6B35',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#FF6B35', alpha=0.9))

# Formatting
ax1.set_xlabel('Stage 1 (Duration) Model Accuracy (%)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Sales per 8-Hour Shift', fontsize=13, fontweight='bold')
ax1.set_title('Accuracy Threshold Analysis:\nWhen Does Efficiency Scheduling Beat Probability-Only?', 
              fontsize=15, fontweight='bold', pad=15)

ax1.set_xlim(55, 102)
ax1.set_ylim(50, 120)
ax1.set_xticks([58, 60, 65, 70, 75, 80, 85, 90, 95, 100])
ax1.grid(True, alpha=0.3, linestyle='-')
ax1.legend(loc='lower right', fontsize=10)

# Add data labels for efficiency line
for i, (x, y) in enumerate(zip(accuracy_pct, df['efficiency_sales'])):
    offset = 8 if y < df['probability_sales'].iloc[0] else -12
    ax1.annotate(f'{y}', xy=(x, y + offset), fontsize=9, ha='center', 
                 fontweight='bold', color=color_eff)

# Add gap annotation
ax1.annotate('', xy=(75, 70), xytext=(57.59, 70),
             arrowprops=dict(arrowstyle='<->', color='#333', lw=1.5))
ax1.text(66, 66, 'Gap: 17.4 pp', fontsize=10, ha='center', style='italic')

plt.tight_layout()
plt.savefig('accuracy_threshold_plot.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('plots/accuracy_threshold_plot.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Plot saved to: accuracy_threshold_plot.png")
print("Plot also saved to: plots/accuracy_threshold_plot.png")

# Also create a secondary plot showing the sales difference
fig2, ax2 = plt.subplots(figsize=(10, 5))

colors = ['#E94F37' if d < 0 else '#2E86AB' for d in df['sales_diff']]
bars = ax2.bar(accuracy_pct, df['sales_diff'], color=colors, edgecolor='white', linewidth=1.5)

ax2.axhline(y=0, color='black', linewidth=1)
ax2.axvline(x=75, color='#333', linewidth=2, linestyle=':', alpha=0.7)
ax2.axvline(x=57.59, color='#FF6B35', linewidth=2, linestyle='-', alpha=0.8)

# Add value labels on bars
for bar, diff in zip(bars, df['sales_diff']):
    height = bar.get_height()
    offset = 2 if height >= 0 else -4
    ax2.text(bar.get_x() + bar.get_width()/2, height + offset,
             f'{int(diff):+d}', ha='center', va='bottom' if height >= 0 else 'top',
             fontsize=11, fontweight='bold')

ax2.set_xlabel('Stage 1 (Duration) Model Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Sales Difference vs Probability Strategy', fontsize=12, fontweight='bold')
ax2.set_title('Efficiency Strategy: Sales Gain/Loss vs Probability-Only Baseline', 
              fontsize=14, fontweight='bold')
ax2.set_xticks(accuracy_pct)
ax2.set_xticklabels([f'{int(x)}%' for x in accuracy_pct])
ax2.grid(True, alpha=0.3, axis='y')

# Add annotations
ax2.annotate('Threshold ~75%', xy=(75, 15), fontsize=10, ha='center')
ax2.annotate(f'Current: {current_acc:.1f}%', xy=(57.59, -18), fontsize=10, 
             ha='center', color='#FF6B35', fontweight='bold')

plt.tight_layout()
plt.savefig('accuracy_threshold_diff_plot.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('plots/accuracy_threshold_diff_plot.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Difference plot saved to: accuracy_threshold_diff_plot.png")
print("Difference plot also saved to: plots/accuracy_threshold_diff_plot.png")
