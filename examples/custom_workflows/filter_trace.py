import argparse
import gzip
import ijson
import json
import logging

from pathlib import Path
from decimal import Decimal
from typing import Any, Optional, Iterable

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
                        prog='filter_trace.py',
                        description='Filter a trace to only include events in the range of a given user annotation',
                        epilog="""Example: To filter a section profiled with torch.autograd.profiler.record_function("my_annotation"), use: python filter_trace.py trace.json.gz --annotation "my_annotation" --output filtered.json.gz""")
    parser.add_argument('filename', type=str, help=".json or .gz trace file to filter")
    parser.add_argument("--annotation", type=str, help="Name of the user annotation to filter by", required=True)
    parser.add_argument("--output", type=str, help="Output file name with .json or .gz extension", default="filtered.json.gz")
    parser.add_argument("--allow-overwrite", action='store_true', help="Allow overwriting the output file if it exists")
    return parser

def find_user_annotation(event_iter: Iterable[dict[str, Any]], user_annotation_name: str) -> Optional[dict[str, Any]]:
    for event in event_iter:
        if event.get('cat', None) != 'user_annotation':
            continue
        if event.get('name', None) == user_annotation_name:
            logger.debug(f"Found user annotation event with name matching {user_annotation_name=}: {event}")
            return event
    return None

def find_user_annotation_in_trace_file(json_file: Path, user_annotation_name: str) -> Optional[dict[str, Any]]:
    with gzip.open(json_file, 'rt') if json_file.suffix == ".gz" else open(json_file, "r", encoding="UTF-8") as f:
        return find_user_annotation(ijson.items(f, "traceEvents.item"), user_annotation_name=user_annotation_name)

def filter_trace_events_to_range(events: Iterable[dict[str, Any]], start: Decimal, end: Decimal) -> list[dict[str, Any]]:
    """Filter trace events to the range of start and end timestamps."""
    filtered_events = []
    for event in events:
        if event['ts'] > end:
            # reject: started after range of interest
            continue
        if 'dur' in event:
            event_end = event['ts'] + event['dur']
            if event_end < start:
                # reject: ended before range of interest
                continue
        # accept: event in range
        filtered_events.append(event)

    return filtered_events

def main(trace_file: Path | str, user_annotation_name: str, output_file: Path | str, allow_overwrite: bool = False) -> None:
    """Read input trace file and filter it to only cover the range of the given user annotation, write output to file."""
    if isinstance(trace_file, str):
        trace_file = Path(trace_file)
    if not trace_file.exists():
        logger.warning(f"Trace file {trace_file.name} does not exist")
        return

    if isinstance(output_file, str):
        output_file = Path(output_file)
    if output_file.exists() and not allow_overwrite:
        logger.warning(f"Output file {output_file.name} already exists. Use --allow-overwrite to overwrite.")
        return

    # Load the json file
    logger.info(f"Loading input trace file {trace_file.name}")
    with gzip.open(trace_file, 'rt') if trace_file.suffix == ".gz" else open(trace_file, "rb") as f:
        data = json.load(f)

    # Find the user annotation event to determine filtering range
    logger.info(f"Finding user annotation with name matching {user_annotation_name}")
    user_annotation_event = find_user_annotation(data["traceEvents"], user_annotation_name=user_annotation_name)
    if user_annotation_event is None:
        logger.warning(f"Could not find event with 'cat': 'user_annotation' and 'name': {user_annotation_name}")
        return

    # Filter events to the range of the user annotation
    logger.debug("Filtering events to range of user annotation")
    start = user_annotation_event['ts']
    end = start + user_annotation_event['dur']
    data["traceEvents"] = filter_trace_events_to_range(data["traceEvents"], start, end)

    # Dump to filtered output
    logger.debug(f"Writing filtered trace to {output_file}")
    with gzip.open(output_file, 'wt', encoding='UTF-8') if output_file.suffix == ".gz" else open(output_file, 'wt', encoding='UTF-8') as fout:
        json.dump(data, fout)


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    main(trace_file=args.filename, user_annotation_name=args.annotation, output_file=args.output)