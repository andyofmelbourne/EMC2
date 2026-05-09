"""
make an ssh class

establish ssh connection
    copy code to code_directory_remote
        set environment variables
            synchronize working_directories
                run code or
                submit slurm script
            monitor terminal output
            monitor job status
"""
import os, sys
import subprocess
from pathlib import Path
import argparse
from .utils import MyFormatter
import textwrap

def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=MyFormatter,
        description="""run emc over ssh connection"""
    )

    parser.add_argument(
        'config',
        type=Path,
        help='python configuration file'
    )

    parser.add_argument(
        'run_emc',
        type=Path,
        help='run_emc python script'
    )

    parser.add_argument(
        '--init',
        action='store_true',
        help='initialise config pickle file on remote'
    )

    args = parser.parse_args()
    return args


class SSH():
    def __init__(self, host, *args, **kwargs):
        self.host = host

    def test_connection(self):
        cmd = ["ssh", f"{self.host}", "echo hello"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.stdout.strip() == 'hello':
            return True
        else:
            return False

    def transfer(self, file_local, file_remote, send=False, args=[]):
        # add host name
        file_remote = f'{self.host}:' + str(file_remote)

        cmd = ["rsync", '-vrtlpzhP', '--mkpath'] + args + [str(file_local), file_remote]

        # send or recieve
        if not send:
            cmd[-2], cmd[-1] = cmd[-1], cmd[-2]

        # result = subprocess.run(cmd, capture_output=True, text=True)
        return self._local_cmd(cmd)

    def send_file(self, file_local, file_remote, args=[]):
        self.transfer(file_local, file_remote, True, args)

    def receive_file(self, file_local, file_remote, args=[]):
        self.transfer(file_local, file_remote, False, args)

    def _remote_cmd(self, cmd):
        print(' '.join(cmd))
        print(cmd)

        p = subprocess.Popen(
            ' '.join(cmd),
            text=True,
            shell=True,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        p.wait()

        if p.returncode != 0:
            raise ValueError('something went wrong with transfer')

        return p.returncode

    def _local_cmd(self, cmd):
        print(' '.join(cmd))
        print(cmd)

        p = subprocess.Popen(
                cmd,
                stdout=sys.stdout,
                stderr=sys.stderr
                )
        p.wait()

        if p.returncode != 0:
            raise ValueError('something went wrong with transfer')

        return p.returncode

    def remote_cmds(self, cmds):
        a = '"' + ' && '.join(cmds) + '"'
        cmd = ['ssh', self.host, a]
        return self._remote_cmd(cmd)


class SSH_emc(SSH):

    def __init__(self,
                 *args,
                 host='',
                 working_directory_remote='',
                 working_directory_local='',
                 code_directory_remote='',
                 code_directory_local='',
                 cxi_file_remote='',
                 cxi_file_local='',
                 conda_env_remote='',
                 **kwargs
                 ):
        super().__init__(host)
        self.wdr = working_directory_remote
        self.working_directory_remote = working_directory_remote
        self.working_directory_local = working_directory_local
        self.code_directory_remote = code_directory_remote
        self.code_directory_local = code_directory_local
        self.cxi_file_remote = cxi_file_remote
        self.cxi_file_local = cxi_file_local
        self.conda_env_remote = conda_env_remote
        self.cxi_file_name = Path(self.cxi_file_remote).name

    def send_code(self):
        args = (
                '--exclude *.swp --exclude .git/ '
                '--exclude tests --exclude .gitignore '
                '--exclude *.pyc --exclude testing'
                )
        args = args.split(' ')

        self.send_file(
                self.code_directory_local,
                self.code_directory_remote,
                args
                )

    def send_cxi(self):
        self.send_file(
                self.cxi_file_local,
                self.cxi_file_remote
                )

    def get_iteration_info(self):
        pr = Path(self.working_directory_remote) / 'iteration_info.h5'
        pl = Path(self.working_directory_local) / 'iteration_info.h5'
        self.receive_file(
                pl,
                pr
                )

    def add_symlink(self):
        p = Path(self.working_directory_remote) / self.cxi_file_name
        cmds = [f'ln -sf {self.cxi_file_remote} {p}']
        self.remote_cmds(cmds)


class SLURM():

    def __init__(
            self,
            *args,
            sbatch_preamble=None,
            job_name=None,
            command=None,
            is_done='finished',
            prepend=[],
            **kwargs
            ):
        working_directory_remote = kwargs['working_directory_remote']

        self.job_name = job_name
        self.log_file = job_name + '_slurm.log'
        self.is_done = is_done

        self.slurm_script = (
                sbatch_preamble + \
                textwrap.dedent(f"""
                #SBATCH -J {job_name}
                #SBATCH -o {working_directory_remote}/{self.log_file}
                #SBATCH -e {working_directory_remote}/{self.log_file}
                """) + \
                'set -e\n' + \
                command + '\n'\
                f'echo {is_done}'
                )

        self.script_name = f'{job_name}.slurm'

        # e.g. for ssh: prepend=['ssh', 'host']
        self.prepend = prepend

        self.write_cmd = prepend + \
                f"'cat > {working_directory_remote}/{self.script_name}'".split()

        self.submit_cmd = prepend + \
                f'"cd {working_directory_remote} && sbatch {self.script_name}"'.split()

        self.is_running_cmd = prepend + 'squeue --name {job_name} -h'.split()

        self.job_state_cmd = prepend + \
                ["sacct", "--name", job_name, "--format=JobID,State", "-n", "-P", "--sort=JobID"]

        self.job_success_cm = prepend + \
                ["tail", "-n", "1", f'{working_directory_remote}/{self.log_file}']

        self.stream_log_cmd = self.prepend + \
                ['timeout', '5m', 'tail', '-f', f'{self.working_directory_remote}/{self.log_file}']

    def write_slurm_script(self):
        cmd = ' '.join(self.write_cmd)

        print(cmd)
        print(self.slurm_script)

        result = subprocess.run(
            cmd,
            input=self.slurm_script,
            text=True,
            shell=True,
            stdout=sys.stdout,
            stderr=sys.stderr
        )

        if result.returncode != 0:
            print("Error writing remote file:", result.stderr)
            return False
        return True

    def job_success(self):
        result = subprocess.run(self.job_success_cm, capture_output=True, text=True)

        if result.stdout.strip() == self.is_done:
            return True
        else:
            return False

    def launch(self):
        self.write_slurm_script()

        cmd = ' '.join(self.submit_cmd)

        p = subprocess.run(
                cmd,
                text=True,
                shell=True,
                stdout=sys.stdout,
                stderr=sys.stderr,
                )

        print(p.stdout)

        if p.returncode != 0:
            raise ValueError(f'something went wrong {cmd}')

    def is_job_running(self):
        result = subprocess.run(
            self.is_running_cmd,
            stdout=subprocess.PIPE,
            text=True
        )
        return bool(result.stdout.strip())

    def stream_log_file(self):
        cmd = self.stream_log_cmd

        result = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            text=True,
        )

        # if result.returncode != 0:
        #     raise ValueError(f'something went wrong {cmd}')
        return result

    def get_last_job_state(self):
        """
        Get the final state of the most recent SLURM job with a given name.
        Works even on older SLURM versions without --sort.
        """
        job_name = self.job_name

        cmd = self.prepend + \
                ["sacct", "--name", job_name, "--format=JobID,State", "-n", "-P"]
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            if not lines:
                return None  # no job found

            # Parse (job_id, state) pairs
            job_entries = []
            for line in lines:
                parts = line.split("|")
                if len(parts) >= 2:
                    job_id_str, state = parts[0], parts[1]
                    # Only keep top-level job IDs (no .batch, .extern, etc.)
                    if "." not in job_id_str:
                        try:
                            job_id = int(job_id_str)
                            job_entries.append((job_id, state))
                        except ValueError:
                            continue

            if not job_entries:
                return None

            # Sort by numeric job ID
            job_entries.sort(key=lambda x: x[0])
            last_job_id, last_state = job_entries[-1]
            return last_job_id, last_state

        except FileNotFoundError:
            raise RuntimeError("sacct command not found. Make sure SLURM is installed.")



class SSH_SLURM_emc(SSH_emc, SLURM):
    def __init__(self, *args, **kwargs):
        kwargs['prepend'] = ['ssh', kwargs['host']]

        SSH_emc.__init__(self, *args, **kwargs)
        SLURM.__init__(self, *args, **kwargs)


if __name__ == '__main__':
    import runpy
    args = get_args()

    # runs script but not __main__ (which makes pickle file)
    config = runpy.run_path(args.config)

    opts = config['ssh']
    opts.update(config['slurm'])

    # add command
    opts['command'] += f'\npython {args.run_emc}'

    ssh = SSH_SLURM_emc(**opts)

    # print(f'{ssh.test_connection()=}')
    """
    ssh.send_code()
    ssh.send_cxi()

    # send all files needed by config.py
    for file in opts['files_needed']:
        ssh.send_file(
                file,
                opts['working_directory_remote'])

    ssh.send_file(
            args.run_emc,
            opts['working_directory_remote'])

    ssh.add_symlink()

    ssh.launch()
    """
    # ssh.stream_log_file()
    print(f'{ssh.get_last_job_state()=}')
    print(f'{ssh.is_job_running()=}')
    print(f'{ssh.job_success()=}')
