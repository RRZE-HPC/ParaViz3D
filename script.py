import otf2
import argparse
from otf2.events import Enter, Leave
from collections import defaultdict

def parse_otf2_trace(trace_file_path, output_file_path):
    """
    Parses an OTF2 trace file to extract MPI call timings for each rank,
    converting raw ticks to seconds.

    Args:
        trace_file_path (str): The path to the input .otf2 trace file.
        output_file_path (str): The path for the output summary file.
    """
    num_ranks = 0
    program_start_time = None
    # Default to 2.4 Giga-tick per second if not found, to prevent division by zero.
    timer_resolution = 2_400_000_000
    
    # Using defaultdict to easily append to lists for each rank
    mpi_calls = defaultdict(list)
    # Use a stack for each rank to correctly handle nested regions
    mpi_entry_stack = defaultdict(list)

    print(f"Reading trace file: {trace_file_path}")

    with otf2.reader.open(trace_file_path) as trace:
        num_ranks = len(trace.definitions.locations)
        # Get the timer resolution (ticks per second) from the trace
        timer_resolution = trace.timer_resolution
        print(f"Found {num_ranks} ranks. Timer resolution: {timer_resolution} ticks/sec.")

        # Iterate through all events in the trace
        for location, event in trace.events:
            # Set the program start time on the very first event
            if program_start_time is None:
                program_start_time = event.time

            # Get the rank ID from the location's reference attribute
            # This is the correct way to get the unique location identifier.
            rank_id = location._ref

            # We are only interested in Enter and Leave events which have a region attribute
            if not isinstance(event, (Enter, Leave)):
                continue

            # Check if the event is for an MPI function
            if event.region.name.startswith("MPI_"):
                if isinstance(event, Enter):
                    # Push the start time onto the stack for this rank
                    mpi_entry_stack[rank_id].append(event.time)

                elif isinstance(event, Leave):
                    # If we have a matching Enter event, pop and record the call
                    if mpi_entry_stack[rank_id]:
                        start_time = mpi_entry_stack[rank_id].pop()
                        end_time = event.time
                        # Store the call with its function name for better analysis if needed
                        mpi_calls[rank_id].append((event.region.name, start_time, end_time))
                    else:
                        print(f"Warning: Found a Leave event for {event.region.name} on Rank {rank_id} without a matching Enter event. Skipping.")


    # Handle empty traces
    if program_start_time is None:
        program_start_time = 0

    print(f"Finished processing trace. Writing to {output_file_path}")

    # Write the collected data to the output file
    with open(output_file_path, 'wb') as f:
        # Write data for each rank
        # Use the number of locations as the upper bound for ranks.
        for rank in range(num_ranks):
            if not mpi_calls[rank]:
                continue

            f.write(f"\n# Rank {rank}\n".encode("utf-8"))
            f.write(b"Function;StartTime;EndTime\n")

            # Sort calls by start time
            sorted_calls = sorted(mpi_calls[rank], key=lambda x: x[1])

            for name, start_ticks, end_ticks in sorted_calls:
                # Make timestamps relative to the program start
                relative_start_ticks = start_ticks - program_start_time
                relative_end_ticks = end_ticks - program_start_time

                # Convert from ticks to seconds using the timer resolution
                start_sec = relative_start_ticks / timer_resolution
                end_sec = relative_end_ticks / timer_resolution

                # Reduce precision to 6 decimal places
                start_sec = round(start_sec, 6)
                end_sec = round(end_sec, 6)
                # skip if <2.4 or >2.7 seconds
                # if start_sec < 2.4 and end_sec < 2.4:
                #     continue
                # if start_sec > 2.7 and end_sec > 2.7:
                #     continue

                # Write data as binary (UTF-8 encoded text lines)
                line = f"{name};{start_sec:.6f};{end_sec:.6f}\n"
                f.write(line.encode("utf-8"))
    
    print("Done.")

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Parse an OTF2 trace and output a summary of MPI calls in seconds."
    )
    parser.add_argument(
        "trace_file",
        type=str,
        help="Path to the .otf2 trace file (e.g., traces.otf2)"
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to the desired output file"
    )

    args = parser.parse_args()

    # Make sure you have the otf2 library installed:
    # pip install otf2
    parse_otf2_trace(args.trace_file, args.output_file)
