"""
Gui with menu at top
tabs underneith
with the cxi_gui being the default tab

Leave it to widgets to decide what the overall logic of the gui is.
"""
import signal
import sys
import pyqtgraph as pg

from PyQt5.QtWidgets import QApplication

from .cxi_gui import CXI_viewer
from .mask_gui import Mask_maker
from .iteration_gui import Iteration_gui
from .window_menu_tabs import MainWindow

def main(directory):
    app = QApplication(sys.argv)
    cxi_viewer = lambda : CXI_viewer(directory)
    mask_maker = lambda : Mask_maker(directory)
    iteration_viewer = lambda : Iteration_gui(directory)

    # Example dictionary
    gui_structure = {
        'cxi_viewer': cxi_viewer,
        'mask_maker': mask_maker,
        'iteration_viewer': iteration_viewer,
    }

    signal.signal(signal.SIGINT, signal.SIG_DFL) # allow Control-C
    pg.setConfigOption('background', pg.mkColor(0.1))

    win = MainWindow()
    win.populate_from_dict(gui_structure)

    # add cxi_viewer by default
    win.add_tab(cxi_viewer(), 'cxi_viewer')

    win.resize(800, 500)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = '/home/andyofmelbourne/Documents/2024/p7927/scratch/saved_hits/'

    main(directory)
