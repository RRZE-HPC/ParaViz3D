# This script extracts trace data from a Score-P trace and outputs it to
# timestamp files for Blender animations.




################################################################################
#                              CLI Parameters                                  #
################################################################################

# Defining parameters for the command line call of the python file

import argparse
import math

parser = argparse.ArgumentParser("This script extracts trace data from a Score-P trace and outputs it to as timestamp files for Blender animations.")
parser.add_argument("--root_folder", type=str, nargs='?', const="./", default="./", \
    help="This should be the relative path from this python script to the root folder containing the Score-P folder where the trace data is. The output files will be in this folder, next to this Score-P folder and separate from its trace contents. The first character should not be a slash, but the last one should. Example: \"raw_data/myprogram/\". Default:\"./\"")
# parser.add_argument("--mode", type=str, nargs='?', const="std", default="extract", \
#     help="This sets which output files will be generated. Use one of the following: read_full, read_rank, read_rank_formatted, read_formatted, extract (to generate Ranks.txt and CompStopAndStart.txt), extract_limited (to generate Ranks.txt, CompStopAndStart-limit.txt), extract_full (to generate Ranks.txt, CompStart.txt, CompStop.txt, CompStopAndStart.txt), or extract_discosim. Default:\"extract\"")
parser.add_argument("--scorep_folder", type=str, nargs='?', const="scorep_traces/", default="scorep_traces/", \
    help="The name of the folder used by Score-P to output the trace files to. Default:\"scorep_traces/\"")
# parser.add_argument("--selected_rank", type=str, nargs='?', const=None, default=None, \
#     help="The selected rank for the rank specific trace data. Default:\"None\"")
parser.add_argument("--cycles_per_frame", type=int, nargs='?', const=100000, default=100000, \
    help="The number of cycles per frame as set in the Blender video generation. Default:\"100000\"")
parser.add_argument("--cycle_maxcount", type=int, nargs='?', const=math.inf, default=math.inf, \
    help="The upper limit for the cycle timestamp in the timestamp output files. Default:\"Infinity\"")
parser.add_argument("--loop_maxcount", type=int, nargs='?', const=math.inf, default=math.inf, \
    help="The maximum number of lines in the timestamp output files. Default:\"Infinity\"")
parser.add_argument("--cpu_frequency", type=int, nargs='?', const=2400000000, default=2400000000, \
    help="The CPU frequency in Hz. Default:\"2400000000\"")
parser.add_argument("--framerate", type=int, nargs='?', const=60, default=60, \
    help="The Blender video's number of frames per second. Default:\"60\"")
parser.add_argument("--fading", type=int, nargs='?', const=1, default=1, \
    help="The Blender video's number of frames for a color switch. Default:\"1\"")
parser.add_argument("--threshold", type=int, nargs='?', const=1000000000, default=1000000000, \
    help="The threshold number of frames for a timejump, i.e. a compression of time in the video output because nothing is happening anyway. Default:\"10000000\"")
args = parser.parse_args()


# Parameters shortened for readability
root_folder = args.root_folder
#mode = args.mode
scorep_folder = args.scorep_folder
#selected_rank = args.selected_rank
cycles_per_frame = args.cycles_per_frame
cycle_maxcount = args.cycle_maxcount
loop_maxcount = args.loop_maxcount
cpu_frequency = args.cpu_frequency
framerate = args.framerate
fading = args.fading
threshold = args.threshold

extract_folder = root_folder+"extract/"
trace_file = root_folder+scorep_folder+'traces.otf2'


# Some coloring functions for the terminal outputs
RED = "\033[91m"
GREEN = "\033[92m"
DARK = "\033[30m"
YELLOW = "\033[33m"
BOLD = "\033[1m"
DARKER = "\033[2m"
UNDERLINE = "\033[4m"
ITALIC = "\033[3m"
OFF = "\033[0m"+"\033[00m"


# Printing the parameters and how they are set for the run
print(YELLOW+"\n")
print("PARAMETERS:")
print("   Outputting files to: "+root_folder)
#print("   Mode: ", mode)
print("   Reading trace files from: ", scorep_folder)
#print("   Selected rank: ", selected_rank)
print("   Cycles per frame: ", cycles_per_frame, " so 1 second in the video is ", framerate * cycles_per_frame / cpu_frequency, " seconds in the run, and 1 second in the run is ", cpu_frequency / (framerate * cycles_per_frame), " seconds in the video")
print("   Cycle upper limit for timestamp files: ", cycle_maxcount)
print("   Maxcount of lines for timestamp files: ", loop_maxcount)
print("   CPU frequency: ", cpu_frequency / 1000000000, "GHz")
print("   Framerate: ", framerate, "fps")
print("   Selected trace file: ", trace_file)
print("   Fading (number of frames for a color change in Blender): ", fading)
print("   Threshold for timejumps: ", threshold)
print(OFF+"\n")




################################################################################
#                   File generating functions for Blender                      #
################################################################################

import numpy as np
import otf2
from otf2.events import *
import time


def extract_nbranks() -> int:
    """
    Extracting the number of ranks and write the number of ranks to Ranks.txt and also returns it
    """

    nb_ranks = 0

    print(BOLD+"Extracting the number of ranks from the trace file."+OFF)
    with otf2.reader.open(trace_file) as trace:
        print(YELLOW+"Reading beginning of trace file to get the number of ranks"+OFF)
        nb_ranks = len(trace.definitions.locations)
        print("Detected number of ranks: ", nb_ranks)

    with open(extract_folder+'Ranks.txt', 'w') as ranks:
        ranks.write(str(nb_ranks))
    print(GREEN+"Extracted to: Ranks.txt\n"+OFF)

    return nb_ranks


def extract_absolute_starting_timestamp() -> int:
    """
    Extracting the timestamp for the program start and returns it
    """

    absolute_starting_timestamp = 0
    
    print(BOLD+"Extracting the absolute starting cycle timestamp from the trace file."+OFF)
    with otf2.reader.open(trace_file) as trace:
        print(YELLOW+"Reading beginning of trace file.", OFF)
        for _, event in trace.events:
            if isinstance(event, Enter):
                absolute_starting_timestamp = event.time
                break
        print("Absolute starting cycle timestamp: ", absolute_starting_timestamp)
    
    return absolute_starting_timestamp


def extract_starting_timestamp(nb_ranks : int, absolute_starting_timestamp : int, starting_event="", first_time_encountered=True, include_starting_event=False):
    """
    Some events at the beginning of the execution can be uninteresting to the user and long, this is typically the case of MPI_Init.
    This function extracts the timestamp at which a certain chosen event starts or stops. By default, the end of the first MPI_Init (or MPI_Init_thread) is taken as starting timestamp.
    Any event before this chosen one will be ignored.
    
    Warning:
    The nature of this function implies that the "CompStopAndStart" file can unadvertently become "CompStartAndStop" and mislead into inverting MPI and non-MPI regions.
    Simple solution : the Leave timestamp should be selected for MPI regions (because we start computing), with include_start_event=False
                      the Enter timestamp should be selected for non-MPI regions, with include_start_event=True
                      Then we ignore every previous event with timestamp < result (or take into account with >=).
                      We can also do the opposite (use Enter for MPI and Leave for non-MPI) if later we ignore with <= instead of < (or accept with > instead of >=)
                      This allows to use the Leave timestamp instead of the Enter one for a computation event with maintaining consistency.
                      A warning will be shown if this this is not respected.
    """
    
    
    start = time.time()
    
    global_starting_timestamp = absolute_starting_timestamp
    rank_starting_timestamp = [0] * nb_ranks

    if starting_event != "":
        print(BOLD+"Extracting the absolute cycle timestamp for ", starting_event," from the trace file."+OFF)
        with otf2.reader.open(trace_file) as trace:
            print(YELLOW+"Reading beginning of trace file.", OFF)
            if first_time_encountered:
                counter_timestamp_left = nb_ranks
                if include_starting_event:
                    if starting_event.startswith("MPI_"):
                        print(RED+"Unless you know what you do, you should not use include_starting_event=True with ", starting_event, ", an MPI event as starting_event. Read the warning of this function."+OFF)
                    for location, event in trace.events:
                        if isinstance(event, Enter) and event.region.name.startswith(starting_event):
                            rank = int(location.group.name[9:len(location.group.name)])
                            if rank_starting_timestamp[rank] == 0:
                                rank_starting_timestamp[rank] = event.time
                                counter_timestamp_left -= 1
                                if counter_timestamp_left == 0:
                                    break
                    print("Starting cycle timestamp for ", starting_event, " included is ", min(rank_starting_timestamp), " (which is at", (min(rank_starting_timestamp) - absolute_starting_timestamp)/cpu_frequency, " seconds in the run).")
                else:
                    if not starting_event.startswith("MPI_"):
                        print(RED+"Unless you know what you do, you should not use include_starting_event=False with ", starting_event, ", a non-MPI event as starting_event. Read the warning of this function."+OFF)
                    for location, event in trace.events:
                        if isinstance(event, Leave) and event.region.name.startswith(starting_event):
                            rank = int(location.group.name[9:len(location.group.name)])
                            if rank_starting_timestamp[rank] == 0:
                                rank_starting_timestamp[rank] = event.time
                                counter_timestamp_left -= 1
                                if counter_timestamp_left == 0:
                                    break
                    print("Starting cycle timestamp for ", starting_event, " excluded is ", min(rank_starting_timestamp), " (which is at", (min(rank_starting_timestamp) - absolute_starting_timestamp)/cpu_frequency, " seconds in the run).")

            else:
                print("You chose first_time_encountered=False, you will thus ignore every event before the last starting_event ", starting_event)
                if include_starting_event:
                    if starting_event.startswith("MPI_"):
                        print(RED+"Unless you know what you do, you should not use include_starting_event=True with ", starting_event, ", an MPI event as starting_event. Read the warning of this function."+OFF)
                    for location, event in trace.events:
                        if isinstance(event, Enter) and event.region.name.startswith(starting_event):
                            rank = int(location.group.name[9:len(location.group.name)])
                            rank_starting_timestamp[rank] = event.time
                    print("Starting cycle timestamp for ", starting_event, " excluded is ", min(rank_starting_timestamp), " (which is at", (min(rank_starting_timestamp) - absolute_starting_timestamp)/cpu_frequency, " seconds in the run).")
                else:
                    if not starting_event.startswith("MPI_"):
                        print(RED+"Unless you know what you do, you should not use include_starting_event=False with ", starting_event, ", a non-MPI event as starting_event. Read the warning of this function."+OFF)
                    for location, event in trace.events:
                        if isinstance(event, Leave) and event.region.name.startswith(starting_event):
                            rank = int(location.group.name[9:len(location.group.name)])
                            rank_starting_timestamp[rank] = event.time
                    print("Starting cycle timestamp for ", starting_event, " excluded is ", min(rank_starting_timestamp), " (which is at", (min(rank_starting_timestamp) - absolute_starting_timestamp)/cpu_frequency, " seconds in the run).")
        global_starting_timestamp = min(rank_starting_timestamp)
    end = time.time()
    print(GREEN+"Done in ", int(math.ceil(end - start)), " seconds.\n"+OFF)
    
    return global_starting_timestamp, rank_starting_timestamp


#def count_enter_leave_events(nb_ranks:int) -> int:
def count_test_ranks_numpy(nb_ranks:int) -> int:
    """
    Extracting the maximum number of ENTER AND LEAVE events per rank to create 
    a pre-allocated memory dedicated to receive the timestamps.
    This is faster because of fewer memory allocations (only one, at creation)
    The size of that array will be depending on that number and on the number of ranks.
    """

    start = time.time()

    maxeventcount = 0

    # Array holding for each rank a number that will be incremented for each of its enter and leave events
    rankcount = np.zeros((nb_ranks), dtype=int)

    # We go through the whole trace file and count the number of enter and leave events for each rank
    with otf2.reader.open(trace_file) as trace:
        print(YELLOW+"Reading trace file: ", trace_file+OFF)
        for location, event in trace.events:
            if isinstance(event, Enter) or isinstance(event, Leave):
                if event.region.name.startswith("MPI_"):
                    rank = int(location.group.name[9:len(location.group.name)])
                    rankcount[rank] += 1

    # We then take the maximum of these numbers
    maxeventcount = np.max(rankcount)
    print("Max number of MPI events for a rank: ", maxeventcount)

    end = time.time()
    print(GREEN+"Done in ", int(math.ceil(end - start)), " seconds.\n"+OFF)

    return maxeventcount

def count_all_events() -> int:
    """
    Extracting the maximum number of ANY type of events per rank to create 
    a pre-allocated memory dedicated to receive the timestamps.
    This is faster because of fewer memory allocations (only one, at creation)
    The size of that array will be depending on that number and on the number of ranks.
    """

    start = time.time()
    
    maxeventcount = 0

    with otf2.reader.open(trace_file) as trace:
        print(YELLOW+"Reading trace file: ", trace_file+OFF)
        for location in trace.definitions.locations:
            if maxeventcount < location.number_of_events:
                maxeventcount = location.number_of_events
    print("Max number of events of any type for a rank: ", maxeventcount)

    end = time.time()
    print(GREEN+"Done in ", int(math.ceil(end - start)), " seconds.\n"+OFF)

    return maxeventcount


def calculate_bytesneeded(number:int) -> int:
    """
    Calculating how many bytes are needed to store a given integer number
    """

    bytesneeded = 1
    maxint = 256
    while number >= maxint:
        bytesneeded += 1
        maxint *= 256
    print("Bytes needed for the maximum timestamp int / maximum offset int: ", bytesneeded)

    return bytesneeded




def extract_trace_realrun_numpy(ignore_start=True, starting_event = "MPI_Init", first_time_encountered: bool = True, include_starting_event: bool = False, fading = 1):
    """
    Reading a trace file from a real run, and outputting the timestamps for every MPI event, for each rank.
    """

# Importing the number of ranks and write the number of ranks to Ranks.txt

    nb_ranks = extract_nbranks()

# Importing the starting cycle timestamp
    
    absolute_starting_timestamp = extract_absolute_starting_timestamp()
    if ignore_start:
        starting_timestamp, rank_starting_timestamp = extract_starting_timestamp(nb_ranks, absolute_starting_timestamp, starting_event, first_time_encountered, include_starting_event)
        starting_timestamp -= (fading + 1)
    else:
        starting_timestamp, rank_starting_timestamp = extract_starting_timestamp(nb_ranks)
    global_starting_timestamp = starting_timestamp
    previous_timestamp = starting_timestamp

    
# Counting the maximum number of events in a single rank

    max_nb_events_for_a_rank = count_test_ranks_numpy(nb_ranks=nb_ranks) #count_max_event_number_per_rank(MPI_only=MPI_only, nb_ranks=nb_ranks)

# Reading the trace file and extracting the timestamps

    start = time.time()

    position_array = np.zeros((nb_ranks), dtype=int)
    timestamps_array = np.zeros((nb_ranks, max_nb_events_for_a_rank), dtype=int)


    loop_maxcount_forallranks = loop_maxcount * nb_ranks
    current_timestamp = starting_timestamp
    previous_timestamp = current_timestamp
    timejump_sum = 0
    timejump_list = []

    with otf2.reader.open(trace_file) as trace:
        print(YELLOW+"Reading trace file"+OFF)
        for location, event in trace.events:
            if isinstance(event, Enter) or isinstance(event, Leave):
                if event.region.name.startswith("MPI_"):
                    rank = int(location.group.name[9:len(location.group.name)])
                    if event.time >= rank_starting_timestamp[rank]:
                        if position_array[rank] >= loop_maxcount_forallranks or timestamps_array[rank][position_array[rank]] >= cycle_maxcount: #loop_maxcount or event.time - starting_timestamp >= cycle_maxcount:
                            break
                        previous_timestamp = current_timestamp
                        current_timestamp = event.time
                        if current_timestamp - previous_timestamp > threshold:
                            timejump = current_timestamp - previous_timestamp - 2400
                            #print("Timejump from ", previous_timestamp, " to ", current_timestamp, " of length ", timejump)
                            timejump_list.append(previous_timestamp - starting_timestamp)
                            timejump_list.append(timejump)
                            timejump_sum += timejump                                
                            starting_timestamp += timejump
                        timestamps_array[rank][position_array[rank]] = event.time - starting_timestamp
                        position_array[rank] += 1


    with open(extract_folder+'TimeJumpList.txt', 'w') as TimeJumpList:
        print("Output file: ", extract_folder+'TimeJumpList.txt')
        TimeJumpList.write(str(global_starting_timestamp - absolute_starting_timestamp) + "\n")
    print("Wrote to: TimeJumpList.txt")

    end = time.time()
    print(GREEN+"Dispatched timestamps from trace file to array in ", int(math.ceil(end - start)), " seconds.\n"+OFF)

    start = time.time()
    with open(extract_folder+'CompStopAndStart.txt', 'w') as CompStopAndStart:
        print("Output file: ", extract_folder+'CompStopAndStart.txt')
        for rank in range(nb_ranks):
            CompStopAndStart.write("0\n")
            for pos in range(position_array[rank]):
                CompStopAndStart.write(str(timestamps_array[rank][pos]) + "\n")
    end = time.time()
    print(GREEN+"Wrote to: CompStopAndStart.txt in ", int(math.ceil(end - start)), " seconds.\n"+OFF)






def overwrite_check(filename = "", filepath = "", filenamelist = [], filepathlist = []):
    if filename != "":
        if os.path.isfile(extract_folder+filename):
            overwrite = input(RED+'File '+filename+' already exists.\nOverwrite? Y = yes, N = no\n'+OFF)
            if overwrite.lower() != 'y':
                print("Ok, exiting.\n")
                exit()
            else:
                print("Ok, let's do it\n")
    elif filepath != "":
        if os.path.isfile(filepath):
            overwrite = input(RED+'File '+filepath+' already exists.\nOverwrite? Y = yes, N = no\n'+OFF)
            if overwrite.lower() != 'y':
                print("Ok, exiting.\n")
                exit()
            else:
                print("Ok, let's do it\n")
    elif filenamelist != []:
        for filename_instance in filenamelist:
            if os.path.isfile(extract_folder+filename_instance):
                overwrite = input(RED+'File '+filename_instance+' already exists.\nOverwrite? Y = yes, N = no\n'+OFF)
                if overwrite.lower() != 'y':
                    print("Ok, exiting.\n")
                    exit()
                else:
                    print("Ok, let's do it\n")
    elif filepathlist != []:
        for filepath_instance in filepathlist:
            if os.path.isfile(filepath_instance):
                overwrite = input(RED+'File '+filepath_instance+' already exists.\nOverwrite? Y = yes, N = no\n'+OFF)
                if overwrite.lower() != 'y':
                    print("Ok, exiting.\n")
                    exit()
                else:
                    print("Ok, let's do it\n")
    else:
        print("No filename or filepath or filelist provided for the overwrite check, exiting.")
        exit()




################################################################################
#                                    Main                                      #
################################################################################

if __name__ == "__main__":

    print("Program start.\n")
    
    import os

# Checking the folder paths and the mode
    if not os.path.exists(root_folder):
        print(RED+"Invalid root folder, please check the path.\n"+OFF)
        print("Exiting\n")
        exit()

    if not os.path.exists(root_folder+scorep_folder):
        print(RED+"Invalid Score-P folder, please check the path.\n"+OFF)
        print("Exiting\n")
        exit()

    if not os.path.exists(extract_folder):
        print(YELLOW+"Creating extract folder."+OFF)
        os.makedirs(extract_folder)

# Protecting the output files from overwriting
    # if mode == "read_full":
    #     overwrite_check(filename="raw_trace_data.txt")
    # elif mode == "read_rank":
    #     overwrite_check(filename="raw_trace_data_rank_"+selected_rank+".txt")
    # elif mode == "read_rank_formatted":
    #     overwrite_check(filename="raw_trace_data_rank_"+selected_rank+"_formatted.txt")
    # elif mode == "read_formatted":
    #     overwrite_check(filename="raw_trace_data_formatted.csv")
    # elif mode == "extract":
    overwrite_check(filename="CompStopAndStart.txt")
    # elif mode == "extract_limited":
    #     overwrite_check(filename="CompStopAndStart-limit.txt")
    # elif mode == "extract_full":
    #     overwrite_check(filename="CompStart.txt")
    #     overwrite_check(filename="CompStop.txt")
    #     overwrite_check(filename="CompStopAndStart.txt")
    # elif mode == "extract_diff":
    #     overwrite_check(filename="CompStopAndStart-diff.txt")

# # Checking the loop_maxcount and cycle_maxcount values
#     if mode == "extract_limited" and loop_maxcount is None and cycle_maxcount is None:
#         print(RED+"Invalid loop_maxcount value, please provide a value.\n"+OFF)
#         print("Exiting\n")
#         exit()

    extract_trace_realrun_numpy()

    print(GREEN+BOLD+"\nProgram ended succesfully.\n"+OFF)
