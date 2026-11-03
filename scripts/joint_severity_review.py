"""Offline joint review of the category-v2 pilot. Original reviewer columns stay intact."""
import argparse
import csv
import io
import os
from pathlib import Path
import sys
import tempfile

from review_severity_pilot import CHOICES

FIELDS = list(CHOICES)


class JointStore:
    def __init__(self, root, images_root):
        self.root, self.images_root = Path(root).resolve(), Path(images_root).resolve()
        self.path = self.root / 'joint_review.csv'
        self.last_bytes = self.path.read_bytes()
        reader = csv.DictReader(io.StringIO(self.last_bytes.decode('utf-8-sig')))
        self.columns = reader.fieldnames or []
        required = {'image_id', 'image_path', 'decision_status', 'decision_reason', 'disagreement_fields'}
        required.update(prefix + f for prefix in ('final_', 'reviewer1_', 'reviewer2_') for f in FIELDS)
        if not required.issubset(self.columns):
            raise ValueError('This CSV does not have the joint-review columns.')
        self.rows = list(reader)
        if not self.rows or len({r['image_id'] for r in self.rows}) != len(self.rows):
            raise ValueError('Empty CSV or duplicate image IDs.')
        for r in self.rows:
            image = (self.images_root / r['image_path']).resolve()
            if not image.is_relative_to(self.images_root / 'images') or not image.is_file():
                raise ValueError('Missing or unsafe image: ' + r['image_id'])
            if r['decision_status'] not in ('pending_joint_review', 'draft', 'agreed', 'unresolved'):
                raise ValueError('Unsupported decision status: ' + r['image_id'])

    def save(self, index, answers, reason, status):
        if status not in ('draft', 'agreed', 'unresolved'):
            raise ValueError('Choose a supported decision status.')
        values = {f: answers.get(f, '') for f in FIELDS}
        for f, value in values.items():
            if value not in ('', *(v for _, v in CHOICES[f])):
                raise ValueError('Unsupported answer for ' + f)
        reason = reason.strip()
        if len(reason) > 3000:
            raise ValueError('Please keep the reason under 3,000 characters.')
        if status in ('agreed', 'unresolved') and not reason:
            raise ValueError('Describe your shared reason, or why the case remains unresolved.')
        if status == 'agreed':
            if not all(values.values()):
                raise ValueError('Answer all five questions before marking Agreed. Not sure is allowed.')
            if values['obstruction_target'] == 'unknown' and values['obstruction'] != 'uncertain':
                raise ValueError('For an unknown access target, choose Not sure for blockage.')
            if values['validity'] != 'valid' and (values['scale'] != 'uncertain' or values['obstruction'] != 'uncertain'):
                raise ValueError('For invalid/uncertain photos, extent and blockage must be Not sure.')
        if self.path.read_bytes() != self.last_bytes:
            raise RuntimeError('The CSV changed in another program. Reopen this app to load it safely.')
        row = self.rows[index] | {'final_' + f: v for f, v in values.items()} | {
            'decision_reason': reason, 'decision_status': status}
        backup = self.path.with_suffix('.before-app.csv')
        if not backup.exists():
            with backup.open('xb') as stream:
                stream.write(self.last_bytes)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile('w', encoding='utf-8-sig', newline='', dir=self.root, suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                writer = csv.DictWriter(stream, fieldnames=self.columns)
                writer.writeheader()
                writer.writerows(row if i == index else saved for i, saved in enumerate(self.rows))
                stream.flush(); os.fsync(stream.fileno())
            saved = temporary.read_bytes()
            os.replace(temporary, self.path)
            self.rows[index], self.last_bytes = row, saved
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()


def create_window(store):
    import tkinter as tk
    from tkinter import ttk, messagebox
    from tkinter.scrolledtext import ScrolledText
    from PIL import Image, ImageOps, ImageTk

    window = tk.Tk()
    window.title('EcoPin — Joint Severity Review')
    window.geometry(f'{min(1450, window.winfo_screenwidth()-60)}x{min(990, window.winfo_screenheight()-80)}')
    window.minsize(1020, 720)
    style = ttk.Style(window); style.theme_use('clam')
    style.configure('.', font=('Segoe UI', 10))
    style.configure('TButton', padding=7)
    window.columnconfigure(0, weight=1); window.rowconfigure(1, weight=1)
    progress = tk.StringVar(); status = tk.StringVar()
    header = ttk.Frame(window, padding=12); header.grid(row=0, column=0, sticky='ew')
    ttk.Label(header, text='Discuss & confirm final labels', font=('Segoe UI', 18, 'bold')).pack(side='left')
    ttk.Label(header, textvariable=progress).pack(side='right')
    body = ttk.Frame(window, padding=(12, 0)); body.grid(row=1, column=0, sticky='nsew')
    body.rowconfigure(0, weight=1); body.columnconfigure(0, weight=1)
    photo = tk.Label(body, bg='#e9edf0', width=1, height=1)
    photo.grid(row=0, column=0, sticky='nsew', padx=(0, 12))
    panel = ttk.Frame(body); panel.grid(row=0, column=1, sticky='nsew'); panel.rowconfigure(0, weight=1)
    canvas = tk.Canvas(panel, width=580, highlightthickness=0)
    canvas.grid(row=0, column=0, sticky='nsew')
    scroll = ttk.Scrollbar(panel, orient='vertical', command=canvas.yview); scroll.grid(row=0, column=1, sticky='ns')
    canvas.configure(yscrollcommand=scroll.set)
    form = ttk.Frame(canvas); item = canvas.create_window((0, 0), window=form, anchor='nw')
    form.bind('<Configure>', lambda _: canvas.configure(scrollregion=canvas.bbox('all')))
    canvas.bind('<Configure>', lambda event: canvas.itemconfigure(item, width=event.width))
    title = tk.StringVar(); ttk.Label(form, textvariable=title, wraplength=550, font=('Segoe UI', 11, 'bold')).pack(anchor='w', pady=(0, 8))
    comparison = ttk.Frame(form); comparison.pack(fill='x')
    sources = []
    for i in range(2):
        group = ttk.LabelFrame(comparison, text=f'Reviewer {i+1} — original answers', padding=6)
        group.grid(row=0, column=i, sticky='nsew', padx=(0, 6)); comparison.columnconfigure(i, weight=1)
        text = ScrolledText(group, width=26, height=12, wrap='word', font=('Segoe UI', 10))
        text.pack(fill='both', expand=True); text.configure(state='disabled'); sources.append(text)
    ttk.Label(form, text='Extent = visible amount / footprint, not flood depth or hazard.\nChoose the issue and access target before extent and blockage.', wraplength=550).pack(anchor='w', pady=10)
    variables = {f: tk.StringVar() for f in FIELDS}
    labels = ['1. Environmental problem visible?', '2. Main issue', '3. Visible amount / extent', '4. Access target', '5. Blockage of that target']
    for (f, options), label in zip(CHOICES.items(), labels):
        group = ttk.LabelFrame(form, text=label, padding=4); group.pack(fill='x', pady=3)
        for i, (caption, value) in enumerate(options):
            ttk.Radiobutton(group, text=caption, value=value, variable=variables[f]).grid(row=i//2, column=i%2, sticky='w', padx=(0, 12), pady=2)
    ttk.Label(form, text='Shared reason / why unresolved (required to finish)').pack(anchor='w', pady=(10, 3))
    reason = tk.Text(form, height=4, wrap='word', font=('Segoe UI', 10)); reason.pack(fill='x')
    footer = ttk.Frame(window, padding=12); footer.grid(row=2, column=0, sticky='ew')
    ttk.Label(footer, textvariable=status, wraplength=1300).pack(anchor='w', pady=(0, 7))
    buttons = ttk.Frame(footer); buttons.pack(fill='x')
    index = next((i for i,r in enumerate(store.rows) if r['decision_status'] not in ('agreed', 'unresolved')), 0)
    loading = False; dirty = False; pending = None; original = None; resize_job = None

    def render():
        nonlocal resize_job
        resize_job = None
        if original is not None:
            preview = ImageOps.contain(original, (max(100, photo.winfo_width()-12), max(100, photo.winfo_height()-12)), Image.Resampling.LANCZOS)
            photo.image = ImageTk.PhotoImage(preview); photo.configure(image=photo.image)

    def resized(_):
        nonlocal resize_job
        if resize_job is not None: window.after_cancel(resize_job)
        resize_job = window.after(120, render)

    def update_progress():
        agreed = sum(r['decision_status'] == 'agreed' for r in store.rows)
        unresolved = sum(r['decision_status'] == 'unresolved' for r in store.rows)
        progress.set(f'{agreed} agreed · {unresolved} unresolved · {len(store.rows)-agreed-unresolved} pending')

    def flush(decision='draft'):
        nonlocal dirty, pending
        if pending is not None: window.after_cancel(pending); pending = None
        if not dirty and decision == 'draft': return True
        try:
            store.save(index, {f: v.get() for f,v in variables.items()}, reason.get('1.0', 'end-1c'), decision)
            dirty = False; update_progress()
            status.set(f'Saved: {decision}. Original reviewer answers are unchanged. Output: joint_review.csv')
            return True
        except (OSError, ValueError, RuntimeError) as error:
            message = 'Close the CSV in Excel and retry. Your unsaved choices remain here.' if isinstance(error, PermissionError) else str(error)
            status.set('NOT SAVED: ' + message)
            if decision != 'draft': messagebox.showwarning('Check the decision', message, parent=window)
            return False

    def changed(*_):
        nonlocal dirty, pending
        if loading: return
        dirty = True
        if pending is not None: window.after_cancel(pending)
        pending = window.after(600, flush)
        status.set('Saving draft… Use Agreed only after both reviewers accept the decision.')

    def text_changed(_):
        if reason.edit_modified(): reason.edit_modified(False); changed()

    def show():
        nonlocal loading, dirty, original
        loading = True; row = store.rows[index]
        title.set(f'{row["image_id"]} · Photo {index+1}/{len(store.rows)} · {row["decision_status"]}\nDiscuss: {row["disagreement_fields"]}')
        for f, variable in variables.items(): variable.set(row['final_' + f])
        reason.delete('1.0', 'end'); reason.insert('1.0', row['decision_reason']); reason.edit_modified(False)
        for i, widget in enumerate(sources, 1):
            text = '\n'.join(f'{f.replace("_", " ").title()}: {row[f"reviewer{i}_{f}"]}' for f in FIELDS)
            text += '\n\nEvidence:\n' + row.get(f'reviewer{i}_visible_evidence', '') + '\n\nNotes:\n' + row.get(f'reviewer{i}_notes', '')
            widget.configure(state='normal'); widget.delete('1.0', 'end'); widget.insert('1.0', text); widget.configure(state='disabled')
        with Image.open(store.images_root / row['image_path']) as image:
            original = ImageOps.exif_transpose(image).convert('RGB'); original.thumbnail((2200, 2200))
        loading = False; dirty = False; update_progress(); canvas.yview_moveto(0); window.after(50, render)
        status.set('Discuss together. Edits autosave as drafts. Not sure and Unresolved are allowed.')

    def move(delta):
        nonlocal index
        if flush(): index = (index + delta) % len(store.rows); show()

    def finish(decision):
        nonlocal index
        if not flush(decision): return
        todo = [i for i in range(index+1, len(store.rows))] + list(range(index+1))
        next_index = next((i for i in todo if store.rows[i]['decision_status'] not in ('agreed', 'unresolved')), None)
        if next_index is None:
            messagebox.showinfo('Joint review saved', f'All {len(store.rows)} cases have a decision.\nReturn joint_review.csv from:\n{store.root}\n\nUnresolved cases still need follow-up.', parent=window)
        else: index = next_index; show()

    def help_window():
        guide = tk.Toplevel(window); guide.title('Severity criteria'); guide.geometry('820x650')
        text = ScrolledText(guide, wrap='word', font=('Segoe UI', 11), padx=14, pady=14); text.pack(fill='both', expand=True)
        text.insert('1.0', (store.images_root/'CRITERIA.md').read_text(encoding='utf-8')); text.configure(state='disabled')

    def close():
        if flush() or messagebox.askyesno('Unsaved choices', 'The latest choices could not save. Close anyway?', parent=window): window.destroy()

    for variable in variables.values(): variable.trace_add('write', changed)
    reason.bind('<<Modified>>', text_changed); photo.bind('<Configure>', resized)
    ttk.Button(buttons, text='Previous', command=lambda: move(-1)).pack(side='left')
    ttk.Button(buttons, text='Next (draft)', command=lambda: move(1)).pack(side='left', padx=6)
    ttk.Button(buttons, text='Criteria', command=help_window).pack(side='left')
    ttk.Button(buttons, text='Original photo', command=lambda: os.startfile(str(store.images_root/store.rows[index]['image_path']))).pack(side='left', padx=6)
    ttk.Button(buttons, text='Agreed & next · Ctrl+Enter', command=lambda: finish('agreed')).pack(side='right')
    ttk.Button(buttons, text='Unresolved & next', command=lambda: finish('unresolved')).pack(side='right', padx=6)
    window.bind('<Control-Return>', lambda _: finish('agreed')); window.protocol('WM_DELETE_WINDOW', close)
    show()
    return window


def main(argv=None):
    location = sys.executable if getattr(sys, 'frozen', False) else __file__
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(location).resolve().parent)
    parser.add_argument('--images-root', type=Path)
    args = parser.parse_args(argv)
    images_root = args.images_root or args.root.parent/'review_pack'
    try:
        if os.name == 'nt':
            import msvcrt
            with (args.root/'.joint-review.lock').open('a+b') as lock:
                lock.seek(0)
                try: msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError: raise RuntimeError('Joint review is already open. Use the existing window.')
                create_window(JointStore(args.root, images_root)).mainloop()
        else:
            create_window(JointStore(args.root, images_root)).mainloop()
    except Exception as error:
        import tkinter as tk
        from tkinter import messagebox
        window = tk.Tk(); window.withdraw(); messagebox.showerror('Cannot open joint review', str(error), parent=window); window.destroy()
        raise


if __name__ == '__main__':
    main()
