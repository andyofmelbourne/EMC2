import sys
from PyQt5.QtWidgets import QApplication, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget
from PyQt5.QtCore import pyqtSignal


class NestedListWidget(QWidget):
    # Custom signal: emits nested path as list of keys
    itemClickedPath = pyqtSignal(list)

    def __init__(self, data=None, parent=None):
        super().__init__(parent)

        self.tree = QTreeWidget(self)
        self.tree.setHeaderHidden(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self.tree)
        self.setLayout(layout)

        if data is not None:
            self.setData(data)

        # Connect click signal
        self.tree.itemClicked.connect(self._on_item_clicked)

    def setData(self, data):
        """Clear and repopulate the tree with new data."""
        self.tree.clear()
        self._add_items(self.tree.invisibleRootItem(), data)

    def _add_items(self, parent_item, data):
        """
        Recursively add groups (dicts) and datasets (strings) to the tree.
        Groups become expandable, datasets are shown as single-line leaves.
        """
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    # subgroup → node
                    item = QTreeWidgetItem([str(key)])
                    parent_item.addChild(item)
                    self._add_items(item, value)
                else:
                    # dataset → already a formatted string
                    item = QTreeWidgetItem([str(value)])
                    parent_item.addChild(item)

    def expandAll(self):
        """Expand all items in the tree."""
        self.tree.expandAll()

    def collapseAll(self):
        """Collapse all items in the tree."""
        self.tree.collapseAll()

    def get_item_path(self, item):
        """
        Build full path (list of keys) from root to item.
        """
        path = []
        while item is not None:
            path.insert(0, item.text(0))  # prepend this item’s name
            item = item.parent()
        return path

    def _on_item_clicked(self, item, column):
        """Handle click events and show full path."""
        path_list = self.get_item_path(item)
        path_str = "/" + "/".join(path_list)

        #print("Nested path:", path_list)
        #print("Clicked:", path_str)
        self.itemClickedPath.emit(path_list)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Example nested data
    nested_data = {
        "Fruits": ["Apple", "Banana", {"Citrus": ["Orange", "Lemon"]}],
        "Vegetables": [
            "Carrot",
            {"Leafy": ["Spinach", "Lettuce"]},
            {"Root": ["Potato", "Beetroot"]}
        ],
        "Dairy": ["Milk", "Cheese"]
    }

    window = NestedListWidget(nested_data)
    window.setWindowTitle("Nested List with Collapsible Sublists")
    window.resize(400, 300)
    window.show()

    sys.exit(app.exec_())

