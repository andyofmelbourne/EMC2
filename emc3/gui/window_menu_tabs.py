"""
Gui with menu at top
tabs underneith
with the cxi_gui being the default tab

Leave it to widgets to decide what the overall logic of the gui is.
"""
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget,
    QVBoxLayout, QLabel, QAction, QMenu, QTabBar
)
from PyQt5.QtCore import Qt


class CustomTabBar(QTabBar):
    """TabBar that supports middle-click to close tabs."""
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            index = self.tabAt(event.pos())
            if index >= 0:
                self.parent().tabCloseRequested.emit(index)
                return
        super().mouseReleaseEvent(event)

class DetachedWindow(QMainWindow):
    def __init__(self, widget, title, on_close_callback):
        super().__init__()
        self.setWindowTitle(title)
        self.setCentralWidget(widget)
        self._on_close_callback = on_close_callback

    def closeEvent(self, event):
        """Reattach the tab instead of destroying the widget."""
        if self._on_close_callback:
            self._on_close_callback(self.centralWidget(), self.windowTitle())
        event.accept()


class DetachableTabWidget(QTabWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Use custom tab bar
        self.setTabBar(CustomTabBar(self))
        self.setTabsClosable(False)   # no close buttons
        self.setMovable(True)
        self.tabCloseRequested.connect(self.close_tab)

        # Store floating windows for detached tabs
        self._detached_windows = []

    def close_tab(self, index):
        widget = self.widget(index)
        if widget:
            widget.deleteLater()
        self.removeTab(index)

    def contextMenuEvent(self, event):
        """Right-click menu on the tab bar."""
        index = self.tabBar().tabAt(event.pos())
        if index < 0:
            return

        menu = QMenu(self)
        detach_action = QAction("Detach Tab", self)
        detach_action.triggered.connect(lambda: self.detach_tab(index))
        menu.addAction(detach_action)
        menu.exec_(event.globalPos())

    def detach_tab(self, index):
        widget = self.widget(index)
        title = self.tabText(index)

        if widget is None:
            return

        self.removeTab(index)

        # Create floating window
        def on_close(widget, title):
            # self.addTab(widget, title)
            self._detached_windows.remove(window)

        window = DetachedWindow(widget, title, on_close)
        window.resize(400, 300)
        window.show()

        # Keep reference
        self._detached_windows.append(window)


class MainWindowOld(QMainWindow):
    def __init__(self):
        super().__init__()

        # Menu bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        new_tab_action = QAction("New Tab", self)
        new_tab_action.triggered.connect(self.add_tab)
        file_menu.addAction(new_tab_action)

        # Use our detachable tab widget
        self.tabs = DetachableTabWidget()
        self.setCentralWidget(self.tabs)

        # Start with one tab
        self.add_tab()

    def add_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(QLabel(f"Hello, I'm tab {self.tabs.count() + 1}!"))
        self.tabs.addTab(widget, f"Tab {self.tabs.count() + 1}")

class MainWindowOld2(QMainWindow):
    def __init__(self):
        super().__init__()

        # Central tab widget
        self.tabs = DetachableTabWidget()
        self.setCentralWidget(self.tabs)

    def populate_from_dict(self, menu_dict):
        """Populate menubar from a nested dictionary."""
        def add_menu_items(menu, structure):
            for key, val in structure.items():
                if isinstance(val, dict):
                    # Submenu
                    submenu = QMenu(key, self)
                    menu.addMenu(submenu)
                    add_menu_items(submenu, val)
                elif isinstance(val, QWidget):
                    # Widget → menu action
                    action = QAction(key, self)
                    action.triggered.connect(lambda _, w=val, t=key: self.add_tab(w, t))
                    menu.addAction(action)
                else:
                    raise TypeError(f"Unsupported value for key {key}: {type(val)}")

        menubar = self.menuBar()
        for top_key, structure in menu_dict.items():
            menu = menubar.addMenu(top_key)
            add_menu_items(menu, structure)

    def add_tab(self, widget, title):
        """Add a widget as a tab."""
        # If widget already has a parent, clone it (widgets can’t be reparented twice easily)
        if widget.parent():
            raise ValueError('added widgets cannot already have a parent')

        self.tabs.addTab(widget, title)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.tabs = DetachableTabWidget()
        self.setCentralWidget(self.tabs)

    def populate_from_dict(self, menu_dict):
        """Populate the menu bar from a nested dict.

        Values can be:
         - dict -> submenu
         - QWidget -> action that opens that widget
         - callable -> factory producing a QWidget (recommended)
        """
        menubar = self.menuBar()

        def add_menu_items(menu, structure):
            for key, val in structure.items():
                if isinstance(val, dict):
                    # Submenu
                    submenu = QMenu(key, self)
                    menu.addMenu(submenu)
                    add_menu_items(submenu, val)
                elif callable(val) or isinstance(val, QWidget):
                    # An action that opens a widget (callable produces a widget)
                    action = QAction(key, self)
                    if callable(val):
                        # call factory each time the action is triggered
                        action.triggered.connect(lambda _=None, f=val, t=key: self.add_tab(f(), t))
                    else:
                        # pass the widget instance; add_tab will clone if needed
                        action.triggered.connect(lambda _=None, w=val, t=key: self.add_tab(w, t))
                    menu.addAction(action)
                else:
                    raise TypeError(f"Unsupported value for key {key}: {type(val)}")

        # Top-level items may be dict (menus) or widget/callable (top-level action)
        for top_key, structure in menu_dict.items():
            if isinstance(structure, dict):
                top_menu = menubar.addMenu(top_key)
                add_menu_items(top_menu, structure)
            elif callable(structure) or isinstance(structure, QWidget):
                action = QAction(top_key, self)
                if callable(structure):
                    action.triggered.connect(lambda _=None, f=structure, t=top_key: self.add_tab(f(), t))
                else:
                    action.triggered.connect(lambda _=None, w=structure, t=top_key: self.add_tab(w, t))
                menubar.addAction(action)
            else:
                raise TypeError(f"Unsupported top-level value for {top_key}: {type(structure)}")

    def add_tab(self, widget, title):
        """Add a widget as a tab. If widget already has a parent, clone it."""
        if widget.parent() is not None:
            # widget = self.clone_widget(widget)
            raise ValueError('added widgets cannot already have a parent')
        self.tabs.addTab(widget, title)
        # Optionally switch to the new tab:
        self.tabs.setCurrentWidget(widget)



if __name__ == "__main__":
    app = QApplication([])

    # Example dictionary
    gui_structure = {
        "New Tab": QWidget, # more than one tab possible
        "File": {
            "New Tab": QWidget(), # only one instance
            "Open": QWidget(),
        },
        "View": {
            "Plots": {
                "1D Plot": QWidget(),
                "2D Plot": QWidget(),
            }
        }
    }

    win = MainWindow()
    win.populate_from_dict(gui_structure)
    win.resize(800, 500)
    win.show()

    app.exec_()
