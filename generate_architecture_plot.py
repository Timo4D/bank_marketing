import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_block(ax, x, y, width, height, text, color='#ddf'):
    rect = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02", 
                                  linewidth=1, edgecolor='black', facecolor=color)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=9, wrap=True)
    return x + width/2, y, y + height # Return connection points (center-x, bottom-y, top-y)

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.5, color='black'))

def main():
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Dimensions
    w = 0.25
    h = 0.06
    
    # 1. Pipeline Start
    # Raw Data
    c1_x, c1_y_bot, c1_y_top = draw_block(ax, 0.375, 0.92, w, h, "Raw Data")
    
    # Preprocessing
    c2_x, c2_y_bot, c2_y_top = draw_block(ax, 0.375, 0.80, w, h, "Preprocessing &\nFeature Engineering")
    draw_arrow(ax, c1_x, c1_y_bot, c2_x, c2_y_top)

    # 2. Split
    # Encoded Features (Left) -> Used in both models
    c3a_x, c3a_y_bot, c3a_y_top = draw_block(ax, 0.15, 0.65, w, h, "Encoded Features (X)", color='#ffcccc')
    # Duration Target (Right)
    c3b_x, c3b_y_bot, c3b_y_top = draw_block(ax, 0.60, 0.65, w, h, "Duration Target\n(Short/Long Median Split)", color='#ccffcc')
    
    draw_arrow(ax, c2_x, c2_y_bot, c3a_x, c3a_y_top+0.01) # Approx connection
    draw_arrow(ax, c2_x, c2_y_bot, c3b_x, c3b_y_top+0.01)

    # 3. Duration Model
    # Takes X and Duration Target
    c4_x, c4_y_bot, c4_y_top = draw_block(ax, 0.60, 0.53, w, h, "Duration Classifier\n(LightGBM)", color='#cce5ff')
    draw_arrow(ax, c3a_x, c3a_y_bot, c4_x, c4_y_top+0.01) # Long arrow from left X
    draw_arrow(ax, c3b_x, c3b_y_bot, c4_x, c4_y_top)

    # Output: Probs
    c5_x, c5_y_bot, c5_y_top = draw_block(ax, 0.60, 0.43, w, h, "Duration Probabilities\n(P_Short, P_Long)", color='#e5ccff')
    draw_arrow(ax, c4_x, c4_y_bot, c5_x, c5_y_top)

    # 4. Feature Enhancement
    # Takes X and Duration Probs
    c6_x, c6_y_bot, c6_y_top = draw_block(ax, 0.375, 0.32, w, h, "Enhanced Feature Set\n(X + Duration Probs)", color='#ffffcc')
    draw_arrow(ax, c3a_x, c3a_y_bot, c6_x, c6_y_top+0.01)
    draw_arrow(ax, c5_x, c5_y_bot, c6_x, c6_y_top+0.01)

    # 5. Outcome Model
    c7_x, c7_y_bot, c7_y_top = draw_block(ax, 0.375, 0.20, w, h, "Outcome Model\n(SMOTE + LightGBM)", color='#ffcc99')
    draw_arrow(ax, c6_x, c6_y_bot, c7_x, c7_y_top)

    # Success Probs
    c8_x, c8_y_bot, c8_y_top = draw_block(ax, 0.375, 0.10, w, h, "Success Probability\n(P_Subscription)", color='#ff9999')
    draw_arrow(ax, c7_x, c7_y_bot, c8_x, c8_y_top)

    # 6. Final Strategy
    c9_x, c9_y_bot, c9_y_top = draw_block(ax, 0.70, 0.10, w, h, "Efficiency Calculator", color='#cccccc')
    # Needs P_Succ (from left) and P_Dur (from way up? or re-use?)
    # Arrow from Success Probs
    draw_arrow(ax, c8_x, c8_y_bot, c9_x, 0.13) # Side entry approx
    # Arrow from Duration Probs (Long arrow down)
    draw_arrow(ax, c5_x, 0.43, c9_x, c9_y_top)

    # Final Output
    c10_x, c10_y_bot, c10_y_top = draw_block(ax, 0.70, 0.02, w, h, "Prioritized Schedule", color='#99ff99')
    draw_arrow(ax, c9_x, c9_y_bot, c10_x, c10_y_top)

    plt.title("Improved Bank Marketing Model Architecture", fontsize=14)
    plt.tight_layout()
    plt.savefig('model_architecture.png', dpi=300, bbox_inches='tight')
    print("Plot saved to model_architecture.png")

if __name__ == "__main__":
    main()
