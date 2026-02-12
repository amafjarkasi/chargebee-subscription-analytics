"""Rich console progress display for the Chargebee mapper."""

import time
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from .client import FetchResult
from .entities import EntityDef, ALL_ENTITIES


class EntityStatus:
    PENDING = "pending"
    FETCHING = "fetching"
    DONE = "done"
    ERROR = "error"
    SKIPPED = "skipped"


class ProgressTracker:
    """Tracks and displays fetch progress using Rich."""

    def __init__(self, console: Console | None = None):
        self.console = console or Console()
        self._entity_states: dict[str, dict[str, Any]] = {}
        self._start_time: float = 0.0
        self._live: Live | None = None
        self._total_api_calls: int = 0

        for entity in ALL_ENTITIES:
            self._entity_states[entity.key] = {
                "name": entity.name,
                "status": EntityStatus.PENDING,
                "records": 0,
                "pages": 0,
                "errors": [],
            }

    def start(self) -> None:
        self._start_time = time.monotonic()
        self._live = Live(
            self._build_table(),
            console=self.console,
            refresh_per_second=4,
        )
        self._live.start()

    def stop(self) -> None:
        if self._live:
            self._live.stop()
            self._live = None

    def _build_table(self) -> Table:
        elapsed = time.monotonic() - self._start_time if self._start_time else 0
        total_records = sum(s["records"] for s in self._entity_states.values())

        table = Table(
            title=f"Chargebee Account Mapper  |  {total_records:,} records  |  {elapsed:.0f}s elapsed",
            show_lines=False,
            pad_edge=True,
            expand=False,
        )
        table.add_column("Entity", style="cyan", min_width=25)
        table.add_column("Status", min_width=10)
        table.add_column("Records", justify="right", min_width=8)
        table.add_column("Pages", justify="right", min_width=6)

        for key in self._entity_states:
            state = self._entity_states[key]
            status = state["status"]

            if status == EntityStatus.PENDING:
                status_text = Text("pending", style="dim")
            elif status == EntityStatus.FETCHING:
                status_text = Text("fetching", style="bold yellow")
            elif status == EntityStatus.DONE:
                status_text = Text("done", style="bold green")
            elif status == EntityStatus.ERROR:
                status_text = Text("error", style="bold red")
            elif status == EntityStatus.SKIPPED:
                status_text = Text("skipped", style="dim yellow")
            else:
                status_text = Text(status)

            records = str(state["records"]) if state["records"] > 0 else "-"
            pages = str(state["pages"]) if state["pages"] > 0 else "-"

            table.add_row(state["name"], status_text, records, pages)

        return table

    def _refresh(self) -> None:
        if self._live:
            self._live.update(self._build_table())

    async def on_entity_start(self, entity: EntityDef) -> None:
        self._entity_states[entity.key]["status"] = EntityStatus.FETCHING
        self._refresh()

    async def on_entity_page(self, entity: EntityDef, page_records: list[dict], page_num: int) -> None:
        state = self._entity_states[entity.key]
        state["records"] += len(page_records)
        state["pages"] = page_num
        self._total_api_calls += 1
        self._refresh()

    async def on_entity_done(self, result: FetchResult) -> None:
        state = self._entity_states[result.entity.key]
        state["records"] = result.count
        state["pages"] = result.pages_fetched
        if result.errors:
            state["status"] = EntityStatus.ERROR
            state["errors"] = result.errors
        elif result.count == 0:
            state["status"] = EntityStatus.SKIPPED
        else:
            state["status"] = EntityStatus.DONE
        self._refresh()

    async def on_entity_error(self, entity: EntityDef, error: str) -> None:
        state = self._entity_states[entity.key]
        state["status"] = EntityStatus.ERROR
        state["errors"].append(error)
        self._refresh()

    def print_summary(self, results: dict[str, FetchResult], elapsed: float) -> None:
        """Print final summary after all fetching is complete."""
        self.console.print()
        self.console.rule("[bold]Fetch Summary")
        self.console.print()

        total_records = sum(r.count for r in results.values())
        total_api_calls = sum(r.api_calls for r in results.values())
        success_count = sum(1 for r in results.values() if r.success and r.count > 0)
        empty_count = sum(1 for r in results.values() if r.success and r.count == 0)
        error_count = sum(1 for r in results.values() if not r.success)

        self.console.print(f"  Total records:    [bold]{total_records:,}[/bold]")
        self.console.print(f"  API calls:        [bold]{total_api_calls:,}[/bold]")
        self.console.print(f"  Time elapsed:     [bold]{elapsed:.1f}s[/bold]")
        self.console.print(f"  Entities fetched: [green]{success_count}[/green]  |  Empty: [yellow]{empty_count}[/yellow]  |  Errors: [red]{error_count}[/red]")

        if error_count > 0:
            self.console.print()
            self.console.print("[bold red]Errors:[/bold red]")
            for key, result in results.items():
                for error in result.errors:
                    self.console.print(f"  [red]- {error}[/red]")

        self.console.print()
