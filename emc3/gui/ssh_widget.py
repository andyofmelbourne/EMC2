import sys
import subprocess
import threading
import runpy
import traceback
from pathlib import Path

from ..ssh import SSH_SLURM_emc

from PyQt5.QtWidgets import (
        QApplication, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
        QLabel, QHBoxLayout, QPlainTextEdit, QPushButton, QMainWindow,
        QLineEdit, QSplitter, QTextEdit
        )
from PyQt5.QtCore import pyqtSignal, QTimer, Qt, QMetaObject, pyqtSlot, QThread, QObject
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen, QTextCursor, QBrush
from PyQt5 import QtCore

class SSHWidget(QWidget):
    """
    run emc over an ssh connection with SLURM

    [load config button] [text input]
    [ssh connection]     [status]
    [last job status]    [status]
    [last job finished]  [status]
    [is running]         [status]
    [sync code button]   [status]
    [sync files button]  [status]
    [run button]         [status]
    [get iteration info] [status]
    [get config.pickle]  [status]

    show slurm log output in window
    [run button]         [status]
    """
    ssh = None
    run_fnam = None
    config_fnam = None

    def __init__(self, parent=None):
        super().__init__(parent)

        # Layout
        Hlayout = QHBoxLayout(self)
        Hlayout.setContentsMargins(0,0,0,0)
        Hlayout.setSpacing(0)

        leftPanel = QWidget()
        layout = QVBoxLayout(leftPanel)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)

        # [load config button] [status] [text input]
        # ------------------------------------------
        # set self.config_fnam on button click
        # load ssh if run_fnam is set
        # ------------------------------------------
        self.config_widget = StatusIndicatorWorker(
                self.load_config,
                label_text="Load config.py",
                button=True,
                line_text='config.py',
                line_edit=True,
                parent=self)
        layout.addWidget(self.config_widget)


        # [load run script button] [text input]
        # ------------------------------------------
        # set self.run_fnam on button click
        # load ssh if config_fnam is set
        # ------------------------------------------
        self.run_widget = StatusIndicatorWorker(
                self.load_run_script,
                label_text="Load run script",
                button=True,
                line_text='run_emc_maxwell.py',
                line_edit=True,
                parent=self)
        layout.addWidget(self.run_widget)

        # [ssh connection] [status]
        # ----------------------------
        self.ssh_connection_widget = StatusIndicatorWorker(
                self.check_connection,
                label_text="ssh connection",
                parent=self)
        layout.addWidget(self.ssh_connection_widget)

        # Timer to poll ssh connection
        self.ssh_timer = QTimer(self)
        self.ssh_timer.timeout.connect(self.ssh_connection_widget.run)
        self.ssh_timer.start(5000)  # check every 3 s

        # [last job status]    [status]
        self.job_status_widget = StatusIndicatorWorker(
                self.job_status,
                label_text="job status",
                line_edit=True,
                parent=self)
        layout.addWidget(self.job_status_widget)
        self.ssh_timer.timeout.connect(self.job_status_widget.run)

        # [last job status]    [status]
        self.job_finished_widget = StatusIndicatorWorker(
                self.job_finished,
                label_text="job finished",
                parent=self)
        layout.addWidget(self.job_finished_widget)
        self.ssh_timer.timeout.connect(self.job_finished_widget.run)

        # [is running]         [status]
        self.is_running_widget = StatusIndicatorWorker(
                self.is_running,
                label_text="is running",
                parent=self)
        layout.addWidget(self.is_running_widget)
        self.ssh_timer.timeout.connect(self.is_running_widget.run)

        # [sync files button]   [status]
        self.sync_files_widget = StatusIndicatorWorker(
                self.sync_files,
                label_text="sync files",
                button=True,
                parent=self)
        layout.addWidget(self.sync_files_widget)

        # [run button]         [status]
        self.submit_widget = StatusIndicatorWorker(
                self.sumbit,
                label_text="submit job",
                button=True,
                parent=self)
        layout.addWidget(self.submit_widget)

        self.submit_widget.worker

        # [get iteration info] [status]
        self.iter_widget = StatusIndicatorWorker(
                self.get_iteration_info,
                label_text="get iteration_info",
                button=True,
                parent=self)
        layout.addWidget(self.iter_widget)

        # [get config.pickle]  [status]
        self.config_pickle_widget = StatusIndicatorWorker(
                self.get_config_pickle,
                label_text="get config pickle",
                button=True,
                parent=self)
        layout.addWidget(self.config_pickle_widget)

        # [stream output]  [status]
        self.stream_log_widget = StatusIndicator(
                label_text="stream log",
                button=True,
                parent=self)
        layout.addWidget(self.stream_log_widget)

        self.log_widget = ProcessingOutputWidget(self)

        self.submit_widget.finished.connect(self.stream_log)
        self.stream_log_widget.label.clicked.connect(self.stream_log)
        self.stream_log_widget.label.clicked.connect(
                lambda : print('hello')
                )
        self.stream_log()

        def f():
            self.stream_log_widget.set_status('green')
            self.stream_log_widget.label.setEnabled(True)

        self.log_widget.worker.finished.connect(f)

        def f():
            self.stream_log_widget.set_status('red')
            self.stream_log_widget.label.setEnabled(True)

        self.log_widget.worker.failed.connect(f)

        splitter = QSplitter()
        splitter.addWidget(leftPanel)
        splitter.addWidget(self.log_widget)
        splitter.setSizes([200, 600])
        splitter.setStretchFactor(0, 0)       # tree widget does NOT stretch
        splitter.setStretchFactor(1, 1)       # plot widget expands/contracts
        self.splitter = splitter
        Hlayout.addWidget(splitter)

    def stream_log(self):
        print(f'stream log {self.ssh=}')
        if self.ssh:
            self.stream_log_widget.set_status('orange')
            self.stream_log_widget.label.setEnabled(False)
            self.log_widget.run_cmd(self.ssh.stream_log_cmd)


    def load_config(self, text):
        self.config_fnam = text

        if self.run_fnam is not None:
            self.load_ssh()

    def load_run_script(self, text):
        self.run_fnam = text

        if self.config_fnam is not None:
            self.load_ssh()

    def load_ssh(self):
        if self.config_fnam is None or self.run_fnam is None:
            raise ValueError('must load config and run script')

        config = runpy.run_path(self.config_fnam)
        opts = config['ssh']
        opts.update(config['slurm'])
        # add command
        opts['command'] += f'\nmpirun -n $SLURM_NTASKS python {self.run_fnam}'
        self.ssh = SSH_SLURM_emc(**opts)
        self.opts = opts

    def check_connection(self):
        if self.ssh and self.ssh.test_connection():
            return True
        else:
            return False

    def job_status(self):
        if self.ssh:
            r = self.ssh.get_last_job_state()
            if r is None:
                return 'not found'
            else:
                return r[1]
        else:
            return False

    def job_finished(self):
        if self.ssh:
            return self.ssh.job_success()
        else:
            return False

    def is_running(self):
        if self.ssh:
            return self.ssh.is_job_running()
        else:
            return False

    def sync_files(self):
        if self.ssh:
            self.ssh.send_code()
            self.ssh.send_cxi()

            # send all files needed by config.py
            for file in self.opts['files_needed']:
                self.ssh.send_file(
                        file,
                        self.opts['working_directory_remote'])

            self.ssh.send_file(
                    self.run_fnam,
                    self.opts['working_directory_remote'])

            self.ssh.add_symlink()
        else:
            return False

    def sumbit(self):
        if self.ssh:
            self.ssh.launch()

        else:
            return False

    def get_iteration_info(self):
        if self.ssh:
            self.ssh.get_iteration_info()
        else:
            return False

    def get_config_pickle(self):
        if self.ssh:
            config_pickle = Path(self.config_fnam).stem + '.pickle'
            pr = Path(self.ssh.working_directory_remote) / config_pickle
            pl = Path(self.ssh.working_directory_local) / config_pickle
            self.ssh.receive_file(
                    pl,
                    pr
                    )
        else:
            return False

class SubWorker(QObject):
    outputReady = pyqtSignal(str)
    finished = pyqtSignal(int)       # returncode
    failed = pyqtSignal(str)         # error message

    def __init__(self):
        super().__init__()
        self._thread = None
        self._process = None

    def run_command(self, cmd):
        """Run a subprocess command in a background thread."""
        # If a previous thread is still running, wait for it to finish
        if self._thread and self._thread.is_alive():
            return  # optionally, you could kill or queue commands

        def target():
            try:
                self._process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )

                # Read stdout
                for line in self._process.stdout:
                    self.outputReady.emit(line.rstrip("\n"))

                # Read stderr
                for line in self._process.stderr:
                    self.outputReady.emit(f"[ERR] {line.rstrip()}")

                self._process.wait()
                self.finished.emit(self._process.returncode)

            except Exception as e:
                self.failed.emit(str(e))

        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the current process if running."""
        if self._process and self._process.poll() is None:
            self._process.terminate()
            self._process.wait()

class ProcessingOutputWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        layout = QVBoxLayout(self)

        # Text display
        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)
        layout.addWidget(self.text_edit)

        # Worker
        self.worker = SubWorker()
        self.worker.outputReady.connect(self.append_text)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)

    def run_cmd(self, cmd_list):
        # You could pass any command list here, e.g. from user input
        self.text_edit.clear()
        self.worker.run_command(cmd_list)

    def append_text(self, text: str):
        self.text_edit.append(text)

    def on_finished(self, code: int):
        if code == 0:
            self.text_edit.append(f"\n✅ Process finished successfully (code={code})")
        else:
            self.text_edit.append(f"\n❌ Process exited with code {code}")

    def on_failed(self, error: str):
        self.text_edit.append(f"\n⚠️ Failed to start process: {error}")


class StatusIndicator(QWidget):
    def __init__(self, label_text="Status", button=False,
                 line_edit=False, line_text='', parent=None):
        super().__init__(parent)

        if button:
            self.label = QPushButton(label_text)
        else:
            self.label = QLabel(label_text)

        self.light = QLabel()
        self.light.setFixedSize(16, 16)
        self.light.setScaledContents(True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)
        layout.addWidget(self.light)
        if line_edit:
            self.edit = QLineEdit(line_text)
            layout.addWidget(self.edit)
        layout.addStretch()
        self.set_status(False)  # Default to "bad" / red

    def set_status(self, color='red'):
        """Set the indicator color: green if ok=True, red otherwise."""
        pixmap = QPixmap(self.light.size())
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(QColor(color)))
        painter.setPen(QPen(Qt.black, 1))
        painter.drawEllipse(0, 0, pixmap.width()-1, pixmap.height()-1)
        painter.end()

        self.light.setPixmap(pixmap)

class StatusIndicatorWorker(QWidget):
    """
    shows a button if button is True else a label with label_text
    shows a status light
    shows an input field if line_edit and line_text is not None
    shows an output field if line_edit and line_text is None

    func must be a callable:
        if button then this triggers call to func
        else func can be triggered by calling run()

        when func is started status is orange
        if func finishes status is green
        if func raises an Error status is red

        if func has output and line_edit and line_text is None
        then line_edit shows func output

        if line_edit and line_text is not None then provide
        text as input to func
    """
    finished = QtCore.pyqtSignal(bool)
    def __init__(
            self,
            func,
            label_text="Status",
            button=False,
            line_edit=False,
            line_text=None,
            parent=None):
        super().__init__(parent)

        if button:
            self.label = QPushButton(label_text)
            self.button = True
        else:
            self.label = QLabel(label_text)
            self.button = False

        self.light = QLabel()
        self.light.setFixedSize(16, 16)
        self.light.setScaledContents(True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)
        layout.addWidget(self.light)
        self.output = False
        self.input = False
        if line_edit:
            self.edit = QLineEdit(line_text)
            layout.addWidget(self.edit)
            if line_text:
                self.edit.setReadOnly(False)
                self.input = True
            else:
                self.edit.setReadOnly(True)
                self.output = True
        layout.addStretch()
        self.set_status(False)  # Default to "bad" / red

        self.func = func
        self.worker = Worker()
        self.worker.started.connect(self.started)
        self.worker.finished.connect(self._finished)
        self.worker.failed.connect(self.failed)

        if button:
            self.label.clicked.connect(self.run)

    def run(self):
        if self.input:
            self.worker.run(self.func, self.edit.text())
        else:
            self.worker.run(self.func)

    def started(self):
        self.set_status('orange')
        if self.button:
            self.label.setEnabled(False)

    def failed(self, e):
        self.set_status('red')
        if self.button:
            self.label.setEnabled(True)

        print(e)

    def _finished(self, msg):
        if self.button:
            self.label.setEnabled(True)

        if self.output:
            self.edit.setText(str(msg))

        if msg is False:
            self.set_status('red')
            self.finished.emit(False)
        else:
            self.set_status('green')
            self.finished.emit(True)

    def set_status(self, color='red'):
        """Set the indicator color: green if ok=True, red otherwise."""
        pixmap = QPixmap(self.light.size())
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(QColor(color)))
        painter.setPen(QPen(Qt.black, 1))
        painter.drawEllipse(0, 0, pixmap.width()-1, pixmap.height()-1)
        painter.end()

        self.light.setPixmap(pixmap)


class Worker(QtCore.QObject):
    started = QtCore.pyqtSignal()
    finished = QtCore.pyqtSignal(object)     # emits result of the function
    failed = QtCore.pyqtSignal(str)          # emits traceback string

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread = QtCore.QThread()
        self.moveToThread(self._thread)
        self._thread.start()

        # Store the current function to execute
        self._current_func = None
        self._current_args = ()
        self._current_kwargs = {}

    @QtCore.pyqtSlot()
    def _execute(self):
        """Internal slot to run the stored function."""
        if self._current_func is None:
            return

        self.started.emit()
        try:
            result = self._current_func(*self._current_args, **self._current_kwargs)
            self.finished.emit(result)
        except Exception:
            tb = traceback.format_exc()
            self.failed.emit(tb)

    def run(self, func, *args, **kwargs):
        """
        Schedule a function to be run in the worker thread.
        Can be called repeatedly.
        """
        self._current_func = func
        self._current_args = args
        self._current_kwargs = kwargs

        # Post event to the worker thread
        QtCore.QMetaObject.invokeMethod(self, "_execute", QtCore.Qt.QueuedConnection)

    def stop(self):
        """Stop the worker thread gracefully."""
        self._thread.quit()
        self._thread.wait()


if __name__ == "__main__":
    import sys, signal
    signal.signal(signal.SIGINT, signal.SIG_DFL) # allow Control-C

    app = QApplication([])
    win = QMainWindow()
    W = SSHWidget()
    win.setCentralWidget(W)
    win.show()

    sys.exit(app.exec_())
