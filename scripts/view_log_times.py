import matplotlib.pyplot as plt
import argparse
import numpy as np


"""
find msgs of the form:
    time_stamp_epoch:string:process_id:...:msg (start)
    time_stamp_epoch:string:process_id:...:msg (stop)

e.g.
    1740615880.113077:INFO:745664:init.py:<module>:86: init (start)

then calculate the duration and display in some kind of timeline

if a function writes to the logfile in seperate processes then
we should treat them as seperate

so the msg and the process_id are both required to uniquely identify
a duration for display
key = (msg, process_id)
"""


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Read log file and show time between log entries"
    )

    parser.add_argument(
        'logfile',
        type=str,
        help='configuration file name'
    )

    parser.add_argument(
        '--collapse_pid',
        action='store_true',
        help='if there is no overlap, collapse prosesses with the '
             'same msg onto one line'
    )

    parser.add_argument(
        '--collapse_overlap',
        action='store_true',
        help='if there is overlap, merge prosesses with the '
             'same msg onto one line'
    )

    parser.add_argument(
        '--discard',
        type=float,
        default=0.,
        help='discard messages that cumulatively take less than '
             '<discard> x total time of the total'
    )

    args = parser.parse_args()
    return args


def detect_no_overlap_start_stop(s1, s2):
    # s1 = [start_time, stop_time]
    # or
    # s1 = [[start_0, stop_0], [start_1, stop_1]]
    out = True
    if hasattr(s1[0], '__len__'):
        for s1_i in s1:
            out = out and detect_no_overlap_start_stop(s1_i, s2)
        return out
    elif hasattr(s2[0], '__len__'):
        for s2_i in s2:
            out = out and detect_no_overlap_start_stop(s1, s2_i)
        return out
    else:
        if (
            s2[0] >= s1[1]
            or s2[1] <= s1[0]
        ):
            return True
        else:
            return False


def detect_overlap_start_stop(s1, s2):
    return not detect_no_overlap_start_stop(s1, s2)


def detect_overlap(p1, p2):
    ss1 = np.array([p1.start_time, p1.stop_time]).T
    ss2 = np.array([p2.start_time, p2.stop_time]).T
    overlap = detect_overlap_start_stop(ss1, ss2)

    return overlap


def join(a, b):
    # concatenate a and b into numpy array
    # allow for float a or b
    out = np.concatenate(
        (
            np.atleast_1d(a),
            np.atleast_1d(b)
        )
    )
    return out


def join_processes(ps, k1, k2):
    p1 = ps[k1]
    p2 = ps[k2]
    p1.start_time = join(p1.start_time, p2.start_time)
    p1.stop_time = join(p1.stop_time, p2.stop_time)
    p1.duration = join(p1.duration, p2.duration)
    ps.pop(k2)


def merge_processes(ps, k1, k2):
    """only works for float start and stop times"""
    p1 = ps[k1]
    p2 = ps[k2]

    p1.start_time = min(p1.start_time, p2.start_time)
    p1.stop_time = max(p1.stop_time, p2.stop_time)
    p1.duration = p1.stop_time - p1.start_time
    ps.pop(k2)


def collapse_pid(processes):
    # loop over unique msgs
    msgs = set(k[0] for k in processes.keys())
    for msg in msgs:
        # i = active key number
        for i in range(5):
            # get keys for this msg
            keys = [key for key in processes if key[0] == msg]

            if i >= len(keys):
                break

            # loop over processes with msg and join with active process
            for j in range(len(keys)):
                if (keys[j] in processes and i != j):
                    p1 = processes[keys[i]]
                    p2 = processes[keys[j]]

                    if not detect_overlap(p1, p2):
                        join_processes(processes, keys[i], keys[j])


def collapse_overlap(processes):
    # loop over unique msgs
    msgs = set(k[0] for k in processes.keys())
    for msg in msgs:
        # i = active key number
        for i in range(5):
            # get keys for this msg
            keys = [key for key in processes if key[0] == msg]

            if i >= len(keys):
                break

            # loop over processes with msg and join with active process
            for j in range(len(keys)):
                if (keys[j] in processes and i != j):
                    p1 = processes[keys[i]]
                    p2 = processes[keys[j]]

                    if detect_overlap(p1, p2):
                        merge_processes(processes, keys[i], keys[j])


class Process():
    def __init__(self, msg, start_time):
        self.msg = msg
        self.start_time = start_time
        self.stop_time = None

    def set_time_stop(self, time):
        self.stop_time = time
        self.duration = time - self.start_time


if __name__ == "__main__":
    args = get_args()

    # get start time of log
    t0 = None
    t1 = None

    processes = {}
    numbers = {}
    with open(args.logfile, 'r') as f:
        for line in f.readlines():
            if t0 is None:
                t0 = float(line.split(':')[0])

            line = line.strip()
            start = line[-7:] == '(start)'
            stop = line[-6:] == '(stop)'

            if start or stop:
                s = line.split(':')

                if start:
                    msg = s[-1][:-7].strip()
                elif stop:
                    msg = s[-1][:-6].strip()

                time = float(s[0]) - t0
                process_id = int(s[2])

                key = (msg, process_id)

                # add msgs
                if start and (key not in processes):
                    processes[key] = Process(msg, time)

                # add repeate msgs
                elif start and (key in processes):
                    if processes[key].stop_time is None:
                        err = f'a duplicate process has started before the '\
                              f'last ono finished!\n{line}'
                        raise ValueError(err)
                    else:
                        # move old process
                        if key not in numbers:
                            numbers[key] = 0
                        new_key = key + (numbers[key],)
                        numbers[key] += 1

                        processes[new_key] = processes[key]

                        # and new process
                        processes[key] = Process(msg, time)

                # add stop time
                elif stop and (key in processes):
                    processes[key].set_time_stop(time)

                # somethings gone wrong
                elif start and (key in processes):
                    err = f'the same message has been used twice!\n{line}'
                    raise ValueError(err)

                elif stop and (key not in processes):
                    err = f'found "msg (stop)" with no corresponding start '\
                          f'message !\n{line}'
                    raise ValueError(err)

                else:
                    err = '?'
                    raise ValueError(err)

        t1 = float(line.split(':')[0])

    # sometimes we lose logs (??)
    # so check if we have start/stop for each one
    keys = processes.keys()
    processes = {k: v for k, v in processes.items() if v.stop_time is not None}

    # show a bar plot
    # | msg ---------- [ random colour ]
    # |
    # time (down)
    if args.collapse_overlap:
        collapse_overlap(processes)

    if args.collapse_pid:
        collapse_pid(processes)

    if args.discard:
        keys = list(processes.keys())
        total_time = t1 - t0
        for k in keys:
            t = np.sum(processes[k].duration)
            if t < args.discard * total_time:
                processes.pop(k)

    # sort process by start time
    keys = list(processes.keys())
    start_times = [np.atleast_1d(processes[k].start_time).min() for k in keys]
    keys_sorted = [keys[i] for i in np.argsort(start_times)]
    processes_sorted = [processes[k] for k in keys_sorted]

    # show messages vs time
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    # assign a unique colour for each msg
    colours = {}

    height = 0
    for process in processes_sorted:
        if process.msg in colours:
            colour = colours[process.msg]
        else:
            colour = None

        rects = ax.barh(
            height,
            process.duration,
            left=process.start_time,
            color=colour
        )

        colours[process.msg] = rects.patches[0].get_facecolor()

        height += 1

    ax.set_yticks(range(0, height))
    ax.set_yticklabels(process.msg for process in processes_sorted)

    ax.set_xlabel('time (s)')
    ax.set_axisbelow(True)  # show grid under bar elements
    ax.grid(which='major', axis='y', linestyle='--')
    # ax.set_xlim(0, times.max())
    plt.tight_layout()
    plt.show()
