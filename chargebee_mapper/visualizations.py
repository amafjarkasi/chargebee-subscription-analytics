"""Visualization module for churn analysis results.

Generates matplotlib/seaborn charts for churn risk analysis.
"""

import logging
from pathlib import Path

logger = logging.getLogger("chargebee_mapper.visualizations")

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not installed - visualizations will be disabled")

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    logger.warning("seaborn not installed - some visualizations may be limited")


def check_visualization_deps() -> bool:
    """Check if visualization dependencies are available."""
    return HAS_MATPLOTLIB


def generate_churn_visualizations(result: "ChurnAnalysisResult", output_dir: Path) -> list[Path]:
    """Generate all churn analysis visualizations.
    
    Returns list of generated file paths.
    """
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib is required for visualizations. Install with: pip install matplotlib seaborn")
        return []
    
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files: list[Path] = []
    
    # Set style
    if HAS_SEABORN:
        sns.set_theme(style="whitegrid", palette="husl")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["figure.dpi"] = 100
    
    try:
        # 1. Risk Distribution Pie Chart
        path = _generate_risk_distribution_chart(result, output_dir)
        if path:
            generated_files.append(path)
        
        # 2. Risk Score Histogram
        path = _generate_score_histogram(result, output_dir)
        if path:
            generated_files.append(path)
        
        # 3. MRR at Risk by Category
        path = _generate_mrr_at_risk_chart(result, output_dir)
        if path:
            generated_files.append(path)
        
        # 4. Top Churn Signals Bar Chart
        path = _generate_signals_chart(result, output_dir)
        if path:
            generated_files.append(path)
        
        # 5. Customer Risk Scatter Plot (Score vs MRR)
        path = _generate_risk_scatter(result, output_dir)
        if path:
            generated_files.append(path)
        
        logger.info("Generated %d visualization files", len(generated_files))
        
    except Exception as e:
        logger.error("Error generating visualizations: %s", e)
    
    return generated_files


def _generate_risk_distribution_chart(result: "ChurnAnalysisResult", output_dir: Path) -> Path | None:
    """Generate pie chart showing risk level distribution."""
    try:
        fig, ax = plt.subplots(figsize=(8, 8))
        
        risk_dist = result.risk_distribution
        labels = []
        sizes = []
        colors = []
        
        color_map = {
            "low": "#2ecc71",      # Green
            "medium": "#f39c12",   # Orange
            "high": "#e74c3c",     # Red
            "critical": "#8e44ad", # Purple
        }
        
        for level in ["low", "medium", "high", "critical"]:
            count = risk_dist.get(level, 0)
            if count > 0:
                labels.append(f"{level.capitalize()}\n({count})")
                sizes.append(count)
                colors.append(color_map[level])
        
        if not sizes:
            plt.close(fig)
            return None
        
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            explode=[0.02] * len(sizes),
        )
        
        for autotext in autotexts:
            autotext.set_fontsize(11)
            autotext.set_fontweight("bold")
        
        ax.set_title(
            f"Customer Churn Risk Distribution\n(n={result.total_customers_analyzed})",
            fontsize=14,
            fontweight="bold",
        )
        
        output_path = output_dir / "churn_risk_distribution.png"
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        
        logger.info("Generated: %s", output_path.name)
        return output_path
        
    except Exception as e:
        logger.error("Failed to generate risk distribution chart: %s", e)
        return None


def _generate_score_histogram(result: "ChurnAnalysisResult", output_dir: Path) -> Path | None:
    """Generate histogram of churn risk scores."""
    try:
        scores = [c.total_score for c in result.customer_scores]
        
        if not scores:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create histogram with color-coded bins
        bins = [0, 30, 50, 70, 100]
        colors = ["#2ecc71", "#f39c12", "#e74c3c", "#8e44ad"]
        
        n, bin_edges, patches = ax.hist(
            scores,
            bins=20,
            edgecolor="white",
            linewidth=0.5,
        )
        
        # Color bars based on risk level
        for patch in patches:
            x = patch.get_x() + patch.get_width() / 2
            if x < 30:
                patch.set_facecolor("#2ecc71")
            elif x < 50:
                patch.set_facecolor("#f39c12")
            elif x < 70:
                patch.set_facecolor("#e74c3c")
            else:
                patch.set_facecolor("#8e44ad")
        
        # Add vertical lines for thresholds
        for threshold, label, color in [
            (30, "Medium", "#f39c12"),
            (50, "High", "#e74c3c"),
            (70, "Critical", "#8e44ad"),
        ]:
            ax.axvline(x=threshold, color=color, linestyle="--", linewidth=1.5, alpha=0.7)
            ax.text(threshold + 1, ax.get_ylim()[1] * 0.9, label, fontsize=9, color=color)
        
        ax.set_xlabel("Churn Risk Score", fontsize=12)
        ax.set_ylabel("Number of Customers", fontsize=12)
        ax.set_title("Distribution of Churn Risk Scores", fontsize=14, fontweight="bold")
        ax.set_xlim(0, 100)
        
        # Add legend
        legend_patches = [
            mpatches.Patch(color="#2ecc71", label="Low (0-29)"),
            mpatches.Patch(color="#f39c12", label="Medium (30-49)"),
            mpatches.Patch(color="#e74c3c", label="High (50-69)"),
            mpatches.Patch(color="#8e44ad", label="Critical (70+)"),
        ]
        ax.legend(handles=legend_patches, loc="upper right")
        
        output_path = output_dir / "churn_score_histogram.png"
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        
        logger.info("Generated: %s", output_path.name)
        return output_path
        
    except Exception as e:
        logger.error("Failed to generate score histogram: %s", e)
        return None


def _generate_mrr_at_risk_chart(result: "ChurnAnalysisResult", output_dir: Path) -> Path | None:
    """Generate bar chart showing MRR at risk by risk level."""
    try:
        # Calculate MRR by risk level
        mrr_by_level = {"low": 0.0, "medium": 0.0, "high": 0.0, "critical": 0.0}
        
        for customer in result.customer_scores:
            mrr_by_level[customer.risk_level] += customer.total_mrr
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        levels = ["Low", "Medium", "High", "Critical"]
        mrr_values = [
            mrr_by_level["low"],
            mrr_by_level["medium"],
            mrr_by_level["high"],
            mrr_by_level["critical"],
        ]
        colors = ["#2ecc71", "#f39c12", "#e74c3c", "#8e44ad"]
        
        bars = ax.bar(levels, mrr_values, color=colors, edgecolor="white", linewidth=1)
        
        # Add value labels on bars
        for bar, val in zip(bars, mrr_values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(mrr_values) * 0.02,
                    f"${val:,.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight="bold",
                )
        
        ax.set_xlabel("Risk Level", fontsize=12)
        ax.set_ylabel("Monthly Recurring Revenue ($)", fontsize=12)
        ax.set_title("MRR at Risk by Churn Risk Level", fontsize=14, fontweight="bold")
        
        # Add total at risk annotation
        total_at_risk = mrr_by_level["high"] + mrr_by_level["critical"]
        ax.annotate(
            f"Total High/Critical: ${total_at_risk:,.0f}/mo",
            xy=(0.95, 0.95),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        
        output_path = output_dir / "churn_mrr_at_risk.png"
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        
        logger.info("Generated: %s", output_path.name)
        return output_path
        
    except Exception as e:
        logger.error("Failed to generate MRR at risk chart: %s", e)
        return None


def _generate_signals_chart(result: "ChurnAnalysisResult", output_dir: Path) -> Path | None:
    """Generate horizontal bar chart of most common churn signals."""
    try:
        # Count signal occurrences
        signal_counts: dict[str, int] = {}
        
        for customer in result.customer_scores:
            for signal in customer.signals:
                key = signal.signal_type
                signal_counts[key] = signal_counts.get(key, 0) + 1
        
        if not signal_counts:
            return None
        
        # Sort by count and take top 10
        sorted_signals = sorted(signal_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Format labels
        label_map = {
            "payment_failed_recent": "Payment Failed (30d)",
            "payment_failed_multiple": "Multiple Payment Failures",
            "dunning_active": "In Dunning",
            "plan_downgraded": "Plan Downgraded/Cancelled",
            "quantity_reduced": "Quantity Reduced",
            "no_recent_invoice": "No Recent Invoice",
            "renewal_soon_no_activity": "Renewal Soon",
            "new_customer": "New Customer (<90d)",
            "low_lifetime_value": "Low MRR",
            "single_subscription": "Single Subscription",
        }
        
        labels = [label_map.get(s[0], s[0]) for s in sorted_signals]
        counts = [s[1] for s in sorted_signals]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Color by signal category
        colors = []
        for signal_type, _ in sorted_signals:
            if "payment" in signal_type or "dunning" in signal_type:
                colors.append("#e74c3c")  # Red for payment issues
            elif "downgrade" in signal_type or "quantity" in signal_type:
                colors.append("#f39c12")  # Orange for subscription issues
            elif "invoice" in signal_type or "renewal" in signal_type:
                colors.append("#3498db")  # Blue for billing
            else:
                colors.append("#9b59b6")  # Purple for tenure/value
        
        bars = ax.barh(labels, counts, color=colors, edgecolor="white", linewidth=0.5)
        
        # Add count labels
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_width() + max(counts) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                str(count),
                va="center",
                fontsize=10,
            )
        
        ax.set_xlabel("Number of Customers", fontsize=12)
        ax.set_title("Most Common Churn Risk Signals", fontsize=14, fontweight="bold")
        ax.invert_yaxis()  # Highest at top
        
        # Add legend
        legend_patches = [
            mpatches.Patch(color="#e74c3c", label="Payment Issues"),
            mpatches.Patch(color="#f39c12", label="Subscription Changes"),
            mpatches.Patch(color="#3498db", label="Billing Activity"),
            mpatches.Patch(color="#9b59b6", label="Customer Profile"),
        ]
        ax.legend(handles=legend_patches, loc="lower right")
        
        output_path = output_dir / "churn_signals.png"
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        
        logger.info("Generated: %s", output_path.name)
        return output_path
        
    except Exception as e:
        logger.error("Failed to generate signals chart: %s", e)
        return None


def _generate_risk_scatter(result: "ChurnAnalysisResult", output_dir: Path) -> Path | None:
    """Generate scatter plot of risk score vs MRR."""
    try:
        scores = []
        mrrs = []
        colors = []
        
        color_map = {
            "low": "#2ecc71",
            "medium": "#f39c12",
            "high": "#e74c3c",
            "critical": "#8e44ad",
        }
        
        for customer in result.customer_scores:
            if customer.total_mrr > 0:  # Only include paying customers
                scores.append(customer.total_score)
                mrrs.append(customer.total_mrr)
                colors.append(color_map[customer.risk_level])
        
        if not scores:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        scatter = ax.scatter(
            scores,
            mrrs,
            c=colors,
            alpha=0.6,
            s=50,
            edgecolors="white",
            linewidth=0.5,
        )
        
        ax.set_xlabel("Churn Risk Score", fontsize=12)
        ax.set_ylabel("Monthly Recurring Revenue ($)", fontsize=12)
        ax.set_title(
            "Customer Risk Score vs. MRR\n(Prioritize high-risk, high-MRR customers)",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlim(0, 100)
        
        # Add quadrant lines
        median_mrr = sorted(mrrs)[len(mrrs) // 2] if mrrs else 0
        ax.axhline(y=median_mrr, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(x=50, color="gray", linestyle="--", alpha=0.5)
        
        # Annotate quadrants
        ax.text(75, max(mrrs) * 0.95, "HIGH PRIORITY", fontsize=10, ha="center", 
                color="#e74c3c", fontweight="bold")
        ax.text(25, max(mrrs) * 0.95, "Monitor", fontsize=10, ha="center", 
                color="#3498db")
        
        # Add legend
        legend_patches = [
            mpatches.Patch(color="#2ecc71", label="Low Risk"),
            mpatches.Patch(color="#f39c12", label="Medium Risk"),
            mpatches.Patch(color="#e74c3c", label="High Risk"),
            mpatches.Patch(color="#8e44ad", label="Critical Risk"),
        ]
        ax.legend(handles=legend_patches, loc="upper left")
        
        output_path = output_dir / "churn_risk_scatter.png"
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        
        logger.info("Generated: %s", output_path.name)
        return output_path
        
    except Exception as e:
        logger.error("Failed to generate risk scatter plot: %s", e)
        return None


# Import type for type hints
if HAS_MATPLOTLIB:
    from .churn_analyzer import ChurnAnalysisResult
