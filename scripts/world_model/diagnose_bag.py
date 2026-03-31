#!/usr/bin/env python3
import sys
import os
from pathlib import Path

def diagnose_mcap(path: str):
    from mcap.stream_reader import StreamReader
    from mcap.records import Channel, Schema
    print(f"\n--- MCAP Topics: {path} ---")
    channels = {}
    schemas = {}
    with open(path, "rb") as f:
        reader = StreamReader(f, emit_chunks=True)
        it = reader.records
        while True:
            try:
                record = next(it)
            except StopIteration:
                break
            except Exception: # skip errors
                continue
                
            if isinstance(record, Schema):
                schemas[record.id] = record
            elif isinstance(record, Channel):
                if record.topic not in channels:
                    schema_name = schemas[record.schema_id].name if record.schema_id in schemas else "unknown"
                    print(f"  {record.topic} [{record.message_encoding}] (Schema: {schema_name})")
                    channels[record.topic] = record

def diagnose_rosbag(path: str):
    from rosbags.highlevel import AnyReader
    print(f"\n--- ROS1 Topics: {path} ---")
    with AnyReader([Path(path)]) as reader:
        for conn in reader.connections:
            print(f"  {conn.topic} [{conn.msgtype}]")

def main():
    if len(sys.argv) < 2:
        print("Usage: diagnose_bag.py <path_to_bag>")
        return
        
    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"Error: {path} does not exist.")
        return
        
    if path.endswith(".mcap"):
        diagnose_mcap(path)
    elif path.endswith(".bag"):
        diagnose_rosbag(path)
    else:
        print("Unknown extension. Trying both...")
        try: diagnose_rosbag(path)
        except: diagnose_mcap(path)

if __name__ == "__main__":
    main()
