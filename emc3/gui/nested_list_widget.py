import sys
from PyQt5.QtWidgets import QApplication, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget
from PyQt5.QtCore import pyqtSignal, Qt


class NestedListWidget(QWidget):
    # Custom signal: emits nested path as list of keys
    itemClicked = pyqtSignal(object)

    def __init__(self, data=None, parent=None):
        """
        data is a nested dictionary of items
        the items must be convertable to str (have a __repr__ method)
        will emit the item (itemClicked) when clicked
        """
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
        self.sync_tree(self.tree.invisibleRootItem(), data)

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
        self.itemClicked.emit(item.data(0, Qt.UserRole))

    def sync_tree(self, parent_item, data_dict):
        """
        parent_item: Can be the QTreeWidget itself or a QTreeWidgetItem
        data_dict: The nested dictionary to sync
        """
        # 1. Map current children for quick lookup: { "Text": ItemObject }
        current_children = {}
        for i in range(parent_item.childCount()):
            child = parent_item.child(i)
            current_children[child.text(0)] = child

        # 2. Add or Update items from the dictionary
        for key, value in data_dict.items():
            if isinstance(value, dict):
                key_disp = str(key)
            else:
                key_disp = str(value)

            if key in current_children:
                # Item exists, reuse it and recurse if it's a nested dict
                child_item = current_children.pop(key_disp)
            else:
                # Item is new, create it
                child_item = QTreeWidgetItem(parent_item, [key_disp])

            # If value is a dict, sync the next level
            if isinstance(value, dict):
                self.sync_tree(child_item, value)
            else:
                # If it's a leaf node, maybe store the value in UserRole
                child_item.setData(0, Qt.UserRole, value)

        # 3. Prune: Anything left in current_children is no longer in the dict
        for orphaned_item in current_children.values():
            # Remove from parent (widget or item)
            if hasattr(parent_item, "takeTopLevelItem"): # It's the TreeWidget
                index = parent_item.indexOfTopLevelItem(orphaned_item)
                parent_item.takeTopLevelItem(index)
            else: # It's a QTreeWidgetItem
                parent_item.removeChild(orphaned_item)



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

