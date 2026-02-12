"""Entry point for the Chargebee Account Mapper.

Usage:
    python -m chargebee_mapper          # Fetch data from Chargebee API
    python -m chargebee_mapper analyze  # Analyze fetched data for churn risk
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

from rich.console import Console

from .client import ChargebeeClient, AuthenticationError
from .config import load_config
from .fetcher import FetchOrchestrator
from .progress import ProgressTracker
from .storage import StorageManager

logger = logging.getLogger("chargebee_mapper")


def setup_logging(log_file: str, log_level: str) -> None:
    """Configure logging to write to both a file and (optionally) stderr.

    The file handler always logs at DEBUG level for full traceability.
    The root logger level is set to the configured level.
    """
    import os
    os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)

    root_logger = logging.getLogger("chargebee_mapper")
    root_logger.setLevel(logging.DEBUG)  # Capture everything; handlers filter

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d [%(levelname)-5s] %(name)-30s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler: always DEBUG for full audit trail
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Stderr handler: only for WARNING+ (don't interfere with Rich live display)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)
    root_logger.addHandler(stderr_handler)


async def run() -> int:
    """Main async entry point. Returns exit code."""
    console = Console()

    # Load and validate configuration
    config = load_config()

    # Ensure output directory exists before setting up logging
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize logging
    setup_logging(str(config.log_file), config.log_level)

    logger.info("=" * 60)
    logger.info("Chargebee Account Mapper starting")
    logger.info("=" * 60)
    logger.info("Site:            %s", config.site)
    logger.info("API key:         %s...%s", config.api_key[:8], config.api_key[-4:])
    logger.info("Concurrency:     %d", config.max_concurrency)
    logger.info("Rate limit RPM:  %d", config.rate_limit_rpm)
    logger.info("Page size:       %d", config.page_size)
    logger.info("Max retries:     %d", config.max_retries)
    logger.info("Log level:       %s", config.log_level)
    logger.info("Output dir:      %s", config.output_dir)

    console.print()
    console.rule("[bold cyan]Chargebee Account Mapper")
    console.print()
    console.print(f"  Site:            [bold]{config.site}[/bold]")
    console.print(f"  API key:         [dim]{config.api_key[:8]}...{config.api_key[-4:]}[/dim]")
    console.print(f"  Concurrency:     {config.max_concurrency}")
    console.print(f"  Rate limit RPM:  {config.rate_limit_rpm}")
    console.print(f"  Page size:       {config.page_size}")
    console.print(f"  Output dir:      {config.output_dir}")
    console.print(f"  Log file:        {config.log_file}")
    console.print()

    # Initialize components
    client = ChargebeeClient(config)
    tracker = ProgressTracker(console)
    storage = StorageManager(config.output_dir)

    orchestrator = FetchOrchestrator(
        client=client,
        config=config,
        on_entity_start=tracker.on_entity_start,
        on_entity_page=tracker.on_entity_page,
        on_entity_done=tracker.on_entity_done,
        on_entity_error=tracker.on_entity_error,
    )

    # Run the fetch pipeline
    start_time = time.monotonic()
    tracker.start()

    try:
        results = await orchestrator.fetch_all()
    except AuthenticationError as e:
        tracker.stop()
        logger.error("Authentication failed: %s", e)
        console.print(f"\n[bold red]Authentication failed:[/bold red] {e}")
        console.print("Please check your CHARGEBEE_API_KEY and CHARGEBEE_SITE environment variables.")
        return 1
    except KeyboardInterrupt:
        tracker.stop()
        logger.warning("Interrupted by user")
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        return 130
    finally:
        tracker.stop()

    elapsed = time.monotonic() - start_time

    # Print fetch summary
    tracker.print_summary(results, elapsed)

    total_records = sum(r.count for r in results.values())
    total_api_calls = sum(r.api_calls for r in results.values())
    success_count = sum(1 for r in results.values() if r.success and r.count > 0)
    error_count = sum(1 for r in results.values() if not r.success)
    logger.info("Fetch summary: %d records, %d API calls, %.1fs elapsed", total_records, total_api_calls, elapsed)
    logger.info("Entities: %d with data, %d with errors", success_count, error_count)

    # Write data to storage
    console.print("[bold]Writing output files...[/bold]")
    logger.info("Writing output files...")
    storage.open()
    try:
        stats = storage.write_all(results, elapsed)
    finally:
        storage.close()

    # Print storage summary
    json_count = len(stats.get("json_files", []))
    sqlite_count = len(stats.get("sqlite_tables", []))
    console.print(f"  JSON files written:   [green]{json_count}[/green]")
    console.print(f"  SQLite tables:        [green]{sqlite_count}[/green]")

    if stats.get("summary_file"):
        console.print(f"  Summary:              {stats['summary_file']}")

    storage_errors = stats.get("errors", [])
    if storage_errors:
        console.print()
        console.print("[bold red]Storage errors:[/bold red]")
        for err in storage_errors:
            console.print(f"  [red]- {err}[/red]")

    console.print()
    console.print(f"[bold green]Done![/bold green] Output saved to [bold]{config.output_dir}[/bold]")
    console.print(f"[dim]Full log: {config.log_file}[/dim]")
    console.print()

    logger.info("Run complete. Output saved to %s", config.output_dir)
    logger.info("=" * 60)

    return 0


ANALYSIS_TYPES = [
    "all",
    "churn",
    "revenue",
    "clv",
    "payment",
    "segmentation",
    "expansion",
    "refund",
    "anomaly",
    "coupon",
    "cohort",
    "events",
    "deduplication",
]


def main() -> None:
    """Synchronous entry point."""
    parser = argparse.ArgumentParser(
        description="Chargebee Account Mapper - Fetch and analyze Chargebee data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Analysis types:
  all             Run all analyses
  churn           Churn risk prediction
  revenue         Revenue forecasting (MRR/ARR)
  clv             Customer lifetime value prediction
  payment         Payment failure risk scoring
  segmentation    Customer segmentation (RFM)
  expansion       Expansion/upsell prediction
  refund          Refund and credit note analysis
  anomaly         Billing anomaly detection
  coupon          Coupon effectiveness analysis
  cohort          Cohort retention analysis
  events          Event sequence pattern mining
  deduplication   Duplicate record detection
""",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Fetch command (default behavior)
    subparsers.add_parser("fetch", help="Fetch data from Chargebee API")
    
    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze", 
        help="Analyze fetched data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    analyze_parser.add_argument(
        "type",
        nargs="?",
        default="all",
        choices=ANALYSIS_TYPES,
        help="Type of analysis to run (default: all)",
    )
    analyze_parser.add_argument(
        "--data-dir",
        type=str,
        help="Directory containing fetched data (default: uses CHARGEBEE_OUTPUT_DIR or ./output)",
    )
    
    args = parser.parse_args()
    
    try:
        if args.command == "analyze":
            exit_code = run_analyze(args)
        else:
            # Default to fetch (including when no command specified)
            exit_code = asyncio.run(run())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        exit_code = 130
    
    sys.exit(exit_code)


def run_analyze(args: argparse.Namespace) -> int:
    """Run analysis on fetched data."""
    console = Console()
    
    # Determine data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        config = load_config()
        data_dir = config.output_dir
    
    # Setup minimal logging for analyze mode
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    
    analysis_type = getattr(args, "type", "all")
    
    console.print()
    console.rule("[bold cyan]Chargebee Data Analysis")
    console.print()
    console.print(f"  Data directory:  [bold]{data_dir}[/bold]")
    console.print(f"  Analysis type:   [bold]{analysis_type}[/bold]")
    console.print()
    
    output_dir = data_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results: dict[str, dict] = {}
    errors: list[str] = []
    
    # Define which analyses to run
    if analysis_type == "all":
        analyses_to_run = [
            "churn", "revenue", "clv", "payment", "segmentation",
            "expansion", "refund", "anomaly", "coupon", "cohort", "events",
            "deduplication"
        ]
    else:
        analyses_to_run = [analysis_type]
    
    # Run each analysis
    for analysis in analyses_to_run:
        console.print(f"[bold]Running {analysis} analysis...[/bold]")
        try:
            result = _run_single_analysis(analysis, data_dir, output_dir, console)
            results[analysis] = result
            console.print(f"  [green]OK[/green] {analysis} complete")
        except Exception as e:
            errors.append(f"{analysis}: {e}")
            console.print(f"  [red]FAIL[/red] {analysis} failed: {e}")
    
    # Print summary
    console.print()
    console.rule("[bold cyan]Analysis Summary")
    console.print()
    
    console.print(f"  Analyses completed: [green]{len(results)}[/green]")
    if errors:
        console.print(f"  Analyses failed:    [red]{len(errors)}[/red]")
    console.print()
    
    # Print key metrics from each analysis
    for analysis, result in results.items():
        _print_analysis_summary(console, analysis, result)
    
    console.print()
    console.print(f"[bold green]Analysis complete![/bold green] Results saved to [bold]{output_dir}[/bold]")
    console.print()
    
    # List output files
    output_files = list(output_dir.glob("*.json"))
    if output_files:
        console.print("[bold]Output files:[/bold]")
        for f in sorted(output_files):
            console.print(f"  {f.name}")
    
    console.print()
    
    return 0 if not errors else 1


def _run_single_analysis(
    analysis: str, data_dir: Path, output_dir: Path, console: Console
) -> dict:
    """Run a single analysis type and return summary dict."""
    
    if analysis == "churn":
        from .churn_analyzer import ChurnAnalyzer
        
        analyzer = ChurnAnalyzer(data_dir, prediction_window_days=90)
        if not analyzer.load_data():
            raise RuntimeError("Failed to load data")
        result = analyzer.analyze()
        analyzer.save_results(result, output_dir)
        
        # Skip visualizations for now (can cause segfault on some platforms)
        # from .visualizations import check_visualization_deps, generate_churn_visualizations
        # if check_visualization_deps():
        #     generate_churn_visualizations(result, output_dir)
        
        return {
            "customers_analyzed": result.total_customers_analyzed,
            "at_risk": result.customers_at_risk,
            "mrr_at_risk": result.total_mrr_at_risk,
        }
    
    elif analysis == "revenue":
        from .revenue_forecaster import RevenueForecaster
        
        analyzer = RevenueForecaster(data_dir, forecast_months=6)
        if not analyzer.load_data():
            raise RuntimeError("Failed to load data")
        result = analyzer.analyze()
        analyzer.save_results(result, output_dir)
        
        return {
            "current_mrr": result.current_mrr,
            "current_arr": result.current_arr,
            "trend": result.trend_direction,
            "growth_monthly": result.growth_rate_monthly,
        }
    
    elif analysis == "clv":
        from .clv_predictor import CLVPredictor
        
        analyzer = CLVPredictor(data_dir)
        if not analyzer.load_data():
            raise RuntimeError("Failed to load data")
        result = analyzer.analyze()
        analyzer.save_results(result, output_dir)
        
        return {
            "total_customers": result.total_customers,
            "avg_clv": result.avg_clv,
            "total_predicted_clv": result.total_predicted_clv,
        }
    
    elif analysis == "payment":
        from .payment_failure_predictor import PaymentFailurePredictor
        
        analyzer = PaymentFailurePredictor(data_dir)
        if not analyzer.load_data():
            raise RuntimeError("Failed to load data")
        result = analyzer.analyze()
        analyzer.save_results(result, output_dir)
        
        return {
            "customers_analyzed": result.total_customers_analyzed,
            "at_risk": result.customers_at_risk,
            "failure_rate": result.overall_failure_rate,
        }
    
    elif analysis == "segmentation":
        from .customer_segmentation import CustomerSegmenter
        
        analyzer = CustomerSegmenter(data_dir)
        if not analyzer.load_data():
            raise RuntimeError("Failed to load data")
        result = analyzer.analyze()
        analyzer.save_results(result, output_dir)
        
        return {
            "total_customers": result.total_customers,
            "segments": len(result.segment_distribution),
            "top_segment": max(result.segment_distribution.items(), key=lambda x: x[1])[0] if result.segment_distribution else None,
        }
    
    elif analysis == "expansion":
        from .expansion_predictor import ExpansionPredictor
        
        analyzer = ExpansionPredictor(data_dir)
        if not analyzer.load_data():
            raise RuntimeError("Failed to load data")
        result = analyzer.analyze()
        analyzer.save_results(result, output_dir)
        
        return {
            "customers_analyzed": result.total_customers_analyzed,
            "expansion_candidates": result.expansion_candidates,
            "potential_uplift": result.total_potential_uplift,
        }
    
    elif analysis == "refund":
        from .refund_analyzer import RefundAnalyzer
        
        analyzer = RefundAnalyzer(data_dir)
        analyzer.load_data()
        result = analyzer.analyze()
        analyzer.save_results(result, output_dir)
        
        return {
            "credit_notes": result.total_credit_notes,
            "refund_amount": result.total_refund_amount,
            "refund_rate": result.overall_refund_rate,
        }
    
    elif analysis == "anomaly":
        from .anomaly_detector import AnomalyDetector
        
        analyzer = AnomalyDetector(data_dir)
        analyzer.load_data()
        result = analyzer.analyze()
        analyzer.save_results(result, output_dir)
        
        return {
            "records_analyzed": result.total_records_analyzed,
            "anomalies_detected": result.total_anomalies_detected,
            "critical": result.anomalies_by_severity.get("critical", 0),
        }
    
    elif analysis == "coupon":
        from .coupon_analyzer import CouponAnalyzer
        
        analyzer = CouponAnalyzer(data_dir)
        analyzer.load_data()
        result = analyzer.analyze()
        analyzer.save_results(result, output_dir)
        
        return {
            "total_coupons": result.total_coupons,
            "total_redemptions": result.total_redemptions,
            "overall_roi": result.summary.overall_roi,
        }
    
    elif analysis == "cohort":
        from .cohort_analyzer import CohortAnalyzer
        
        analyzer = CohortAnalyzer(data_dir)
        if not analyzer.load_data():
            raise RuntimeError("Failed to load data")
        result = analyzer.analyze()
        analyzer.save_results(result, output_dir)
        
        return {
            "total_cohorts": result.total_cohorts,
            "total_customers": result.total_customers,
            "avg_month1_retention": result.avg_retention_by_month.get(1, 0),
        }
    
    elif analysis == "events":
        from .event_sequence_analyzer import EventSequenceAnalyzer
        
        analyzer = EventSequenceAnalyzer(data_dir)
        analyzer.load_data()
        result = analyzer.analyze()
        analyzer.save_results(result, output_dir)
        
        return {
            "total_events": result.total_events,
            "customers_with_events": result.total_customers_with_events,
            "patterns_found": len(result.common_patterns),
        }
    
    elif analysis == "deduplication":
        from .deduplication import DeduplicationAnalyzer
        
        analyzer = DeduplicationAnalyzer(data_dir)
        if not analyzer.load_data():
            raise RuntimeError("Failed to load data")
        result = analyzer.analyze()
        analyzer.save_results(result, output_dir)
        
        return {
            "total_records": result.total_records,
            "unique_records": result.unique_records,
            "duplicate_groups": len(result.duplicate_groups),
            "duplication_rate": result.duplication_rate,
        }
    
    else:
        raise ValueError(f"Unknown analysis type: {analysis}")


def _print_analysis_summary(console: Console, analysis: str, result: dict) -> None:
    """Print a brief summary for an analysis result."""
    
    if analysis == "churn":
        console.print(f"  [bold]Churn:[/bold] {result['at_risk']} at risk, ${result['mrr_at_risk']:,.0f} MRR at risk")
    
    elif analysis == "revenue":
        console.print(f"  [bold]Revenue:[/bold] ${result['current_mrr']:,.0f} MRR, {result['trend']} trend, {result['growth_monthly']:.1%}/mo")
    
    elif analysis == "clv":
        console.print(f"  [bold]CLV:[/bold] {result['total_customers']} customers, avg ${result['avg_clv']:,.0f} CLV")
    
    elif analysis == "payment":
        console.print(f"  [bold]Payment:[/bold] {result['at_risk']} at risk, {result['failure_rate']:.1%} failure rate")
    
    elif analysis == "segmentation":
        console.print(f"  [bold]Segmentation:[/bold] {result['total_customers']} customers, top segment: {result['top_segment']}")
    
    elif analysis == "expansion":
        console.print(f"  [bold]Expansion:[/bold] {result['expansion_candidates']} candidates, ${result['potential_uplift']:,.0f} potential")
    
    elif analysis == "refund":
        console.print(f"  [bold]Refund:[/bold] {result['credit_notes']} credit notes, {result['refund_rate']:.1%} rate")
    
    elif analysis == "anomaly":
        console.print(f"  [bold]Anomaly:[/bold] {result['anomalies_detected']} detected ({result['critical']} critical)")
    
    elif analysis == "coupon":
        console.print(f"  [bold]Coupon:[/bold] {result['total_redemptions']} redemptions, {result['overall_roi']:.1f}x ROI")
    
    elif analysis == "cohort":
        console.print(f"  [bold]Cohort:[/bold] {result['total_cohorts']} cohorts, {result['avg_month1_retention']:.0%} M1 retention")
    
    elif analysis == "events":
        console.print(f"  [bold]Events:[/bold] {result['total_events']} events, {result['patterns_found']} patterns found")
    
    elif analysis == "deduplication":
        console.print(f"  [bold]Deduplication:[/bold] {result['duplicate_groups']} duplicate groups, {result['duplication_rate']:.1%} rate")


if __name__ == "__main__":
    main()
