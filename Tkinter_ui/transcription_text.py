from __future__ import annotations

import tkinter as tk


class TranscriptionText(tk.Text):
    """
    Native Tk Text used for transcription output.

    Rendering, RTL/LTR direction and mouse selection remain
    controlled by Tk itself.
    """

    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)

        self.bind(
            "<Control-c>",
            self._copy_selection,
            add="+",
        )

        self.bind(
            "<Control-C>",
            self._copy_selection,
            add="+",
        )

        self.bind(
            "<Control-a>",
            self._select_all,
            add="+",
        )

        self.bind(
            "<Control-A>",
            self._select_all,
            add="+",
        )

    def _copy_selection(self, event=None):
        try:
            start = self.index("sel.first")
            end = self.index("sel.last")
        except tk.TclError:
            return "break"

        try:
            text = self.get(start, end)
        except tk.TclError:
            return "break"

        if text:
            try:
                self.clipboard_clear()
                self.clipboard_append(text)
                self.update()
            except tk.TclError:
                pass

        return "break"

    def _select_all(self, event=None):
        try:
            self.tag_remove("sel", "1.0", "end")
            self.tag_add("sel", "1.0", "end-1c")
            self.mark_set("insert", "end-1c")
            self.see("insert")
        except tk.TclError:
            pass

        return "break"