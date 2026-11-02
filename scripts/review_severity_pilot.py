"""Local, resumable reviewer app for the category-v2 severity pilot. No network access."""
import argparse
import csv
from datetime import datetime
import io
import os
from pathlib import Path
import tempfile

FIELDS = ['image_id', 'image_path', 'criteria_version', 'reviewer_id', 'validity',
          'issue_category', 'scale_basis', 'scale', 'obstruction_target', 'obstruction',
          'visible_evidence', 'notes', 'review_status', 'reviewed_at']
CHOICES = {
    'validity': [('Yes', 'valid'), ('No', 'invalid'), ('Not sure', 'uncertain')],
    'issue_category': [('Waste', 'waste'), ('Flooding', 'flooding'), ('Pollution', 'pollution'), ('Not sure', 'uncertain')],
    'scale': [('Small', 'small'), ('Medium', 'medium'), ('Large', 'large'), ('Not sure', 'uncertain')],
    'obstruction_target': [('Road', 'road'), ('Pedestrian path', 'pedestrian_path'), ('Drain', 'drain'), ('Unknown / not visible', 'unknown')],
    'obstruction': [('None: visibly open', 'none'), ('Partial: passage remains', 'partial'), ('Complete: closed', 'complete'), ('Not sure', 'uncertain')],
}
BASES = {'waste': 'waste accumulation', 'flooding': 'flooded footprint',
         'pollution': 'visible pollution footprint', 'uncertain': 'uncertain', '': ''}


class ReviewStore:
    def __init__(self, root, reviewer):
        if reviewer not in ('reviewer1', 'reviewer2'):
            raise ValueError('Choose reviewer1 or reviewer2.')
        self.root = Path(root).resolve()
        self.output = self.root / f'{reviewer}.csv'
        self.last_bytes = self.output.read_bytes()
        reader = csv.DictReader(io.StringIO(self.last_bytes.decode('utf-8-sig')))
        if reader.fieldnames != FIELDS:
            raise ValueError('This answer sheet has an unsupported format. Original answers were not changed.')
        self.rows = list(reader)
        if not self.rows or len({row['image_id'] for row in self.rows}) != len(self.rows):
            raise ValueError('The answer sheet is empty or has duplicate image IDs.')
        for row in self.rows:
            path = (self.root / row['image_path']).resolve()
            if not path.is_relative_to(self.root / 'images') or not path.is_file():
                raise ValueError(f'Image missing or outside the pack: {row["image_id"]}')

    def save(self, index, answers, complete=False):
        row = dict(self.rows[index])
        for field, choices in CHOICES.items():
            value = answers.get(field, '')
            if value not in ('', *(value for _, value in choices)):
                raise ValueError(f'Choose a supported answer for {field}.')
            row[field] = value
        for field in ('reviewer_id', 'visible_evidence', 'notes'):
            value = answers.get(field, '').strip()
            if len(value) > (100 if field == 'reviewer_id' else 3000):
                raise ValueError(f'{field} is too long.')
            row[field] = value
        if complete:
            required = ['reviewer_id', *CHOICES, 'visible_evidence']
            if any(not row[field] for field in required):
                raise ValueError('Enter your name/initials, answer all five questions, and describe the visible evidence. “Not sure” is allowed.')
            if row['obstruction_target'] == 'unknown' and row['obstruction'] != 'uncertain':
                raise ValueError('If the access target is unknown/not visible, choose “Not sure” for blockage.')
            if row['validity'] != 'valid' and (row['scale'] != 'uncertain' or row['obstruction'] != 'uncertain'):
                raise ValueError('For an invalid or uncertain photo, choose “Not sure” for extent and blockage, and explain why.')
        row['scale_basis'] = BASES[row['issue_category']]
        row['review_status'] = 'complete' if complete else 'draft'
        row['reviewed_at'] = datetime.now().astimezone().isoformat(timespec='seconds') if complete else ''
        if self.output.read_bytes() != self.last_bytes:
            raise RuntimeError('This reviewer sheet changed in another window. Close and reopen this app to load it safely.')
        backup = self.output.with_suffix('.before-app.csv')
        if not backup.exists():
            with backup.open('xb') as stream:
                stream.write(self.last_bytes)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8-sig', newline='',
                                             dir=self.root, suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(row if i == index else saved for i, saved in enumerate(self.rows))
                stream.flush(); os.fsync(stream.fileno())
            saved_bytes = temporary.read_bytes()
            os.replace(temporary, self.output)
            self.last_bytes = saved_bytes
            self.rows[index] = row
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()


def launch(store, reviewer):
    import tkinter as tk
    from tkinter import messagebox, ttk
    from tkinter.scrolledtext import ScrolledText
    from PIL import Image, ImageOps, ImageTk

    window = tk.Tk()
    window.title(f'EcoPin photo review — {reviewer}')
    window.geometry(f'{min(1350, window.winfo_screenwidth()-60)}x{min(940, window.winfo_screenheight()-80)}')
    window.minsize(1020, 740)
    style = ttk.Style(window); style.theme_use('clam')
    style.configure('TLabel', font=('Segoe UI', 10))
    style.configure('TRadiobutton', font=('Segoe UI', 10), padding=3)
    style.configure('TButton', font=('Segoe UI', 10), padding=7)
    style.configure('Primary.TButton', background='#126b5d', foreground='white')
    style.map('Primary.TButton', background=[('active', '#0b5145')])
    window.columnconfigure(0, weight=1); window.rowconfigure(1, weight=1)
    header = ttk.Frame(window, padding=(16, 12)); header.grid(row=0, column=0, sticky='ew')
    progress = tk.StringVar(); status = tk.StringVar(value='Choose your answers. Changes save automatically as drafts.')
    ttk.Label(header, text='Environmental photo review', font=('Segoe UI', 17, 'bold')).pack(side='left')
    ttk.Label(header, textvariable=progress).pack(side='right')
    body = ttk.Frame(window, padding=(16, 0, 16, 8)); body.grid(row=1, column=0, sticky='nsew')
    body.columnconfigure(0, weight=1); body.columnconfigure(1, weight=0); body.rowconfigure(0, weight=1)
    photo_frame = ttk.Frame(body); photo_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 18))
    photo_frame.columnconfigure(0, weight=1); photo_frame.rowconfigure(0, weight=1)
    photo = tk.Label(photo_frame, background='#e8ecea'); photo.grid(row=0, column=0, sticky='nsew')
    photo_id = tk.StringVar(); ttk.Label(photo_frame, textvariable=photo_id, font=('Segoe UI', 12, 'bold')).grid(row=1, column=0, pady=8)
    form_panel = ttk.Frame(body); form_panel.grid(row=0, column=1, sticky='nsew')
    form_panel.rowconfigure(0, weight=1)
    form_canvas = tk.Canvas(form_panel, width=455, highlightthickness=0)
    form_canvas.grid(row=0, column=0, sticky='nsew')
    scroll = ttk.Scrollbar(form_panel, orient='vertical', command=form_canvas.yview)
    scroll.grid(row=0, column=1, sticky='ns'); form_canvas.configure(yscrollcommand=scroll.set)
    form = ttk.Frame(form_canvas)
    form_item = form_canvas.create_window((0,0), window=form, anchor='nw')
    form.bind('<Configure>', lambda _: form_canvas.configure(scrollregion=form_canvas.bbox('all')))
    form_canvas.bind('<Configure>', lambda event: form_canvas.itemconfigure(form_item, width=event.width))
    name = tk.StringVar(value=next((r['reviewer_id'] for r in store.rows if r['reviewer_id']), ''))
    ttk.Label(form, text='Your name or initials').pack(anchor='w')
    ttk.Entry(form, textvariable=name).pack(fill='x', pady=(3, 7))
    variables = {field: tk.StringVar() for field in CHOICES}
    titles = ['1. Is an environmental problem visible?', '2. What is the main issue?',
              '3. What is the visible amount or extent?', '4. Which access are you assessing?',
              '5. How blocked is that access?']
    for (field, choices), title in zip(CHOICES.items(), titles):
        group = ttk.LabelFrame(form, text=title, padding=(6, 3)); group.pack(fill='x', pady=3)
        for i, (label, value) in enumerate(choices):
            ttk.Radiobutton(group, text=label, value=value, variable=variables[field]).grid(row=i//2, column=i%2, sticky='w', padx=(0, 9))
    ttk.Label(form, text='What visible evidence supports your answers? (required)').pack(anchor='w', pady=(7, 2))
    evidence = tk.Text(form, height=3, width=52, wrap='word', font=('Segoe UI', 10)); evidence.pack(fill='x')
    ttk.Label(form, text='Extra notes (optional)').pack(anchor='w', pady=(6, 2))
    notes = tk.Text(form, height=2, width=52, wrap='word', font=('Segoe UI', 10)); notes.pack(fill='x')
    footer = ttk.Frame(window, padding=(16, 4, 16, 12)); footer.grid(row=2, column=0, sticky='ew')
    ttk.Label(footer, textvariable=status, wraplength=1150).pack(anchor='w', pady=(0, 7))
    buttons = ttk.Frame(footer); buttons.pack(fill='x')
    index = next((i for i,r in enumerate(store.rows) if r['review_status'] != 'complete'), 0)
    loading = False; dirty = False; pending = None; current_image = None; resize_job = None

    def answers():
        return {field: variable.get() for field, variable in variables.items()} | {
            'reviewer_id': name.get(), 'visible_evidence': evidence.get('1.0', 'end-1c'), 'notes': notes.get('1.0', 'end-1c')}

    def update_progress():
        done = sum(row['review_status'] == 'complete' for row in store.rows)
        progress.set(f'{done} / {len(store.rows)} completed · {reviewer}')

    def flush(complete=False):
        nonlocal pending, dirty
        if pending is not None: window.after_cancel(pending); pending = None
        if not dirty and not complete: return True
        try:
            store.save(index, answers(), complete=complete)
            dirty = False; update_progress()
            status.set('Saved as complete.' if complete else 'Draft saved automatically. Use Save & next when finished.')
            return True
        except (OSError, ValueError, RuntimeError) as error:
            explanation = f'Close {store.output.name} in Excel or another editor, then try saving again. Your unsaved answers are still here.' if isinstance(error, PermissionError) else str(error)
            status.set(f'NOT SAVED: {explanation}')
            if complete: messagebox.showwarning('Please check your answers', explanation, parent=window)
            return False

    def changed(*_):
        nonlocal pending, dirty
        if loading: return
        dirty = True; status.set('Saving draft…')
        if pending is not None: window.after_cancel(pending)
        pending = window.after(600, flush)

    def text_changed(event):
        if event.widget.edit_modified():
            event.widget.edit_modified(False); changed()

    def render():
        nonlocal resize_job
        resize_job = None
        if current_image is None: return
        preview = ImageOps.contain(current_image, (max(100,photo.winfo_width()-16), max(100,photo.winfo_height()-16)), Image.Resampling.LANCZOS)
        photo.image = ImageTk.PhotoImage(preview); photo.configure(image=photo.image)

    def resized(_):
        nonlocal resize_job
        if resize_job is not None: window.after_cancel(resize_job)
        resize_job = window.after(120, render)

    def show():
        nonlocal loading, dirty, current_image
        loading = True; row = store.rows[index]
        for field, variable in variables.items(): variable.set(row[field])
        for widget, field in [(evidence,'visible_evidence'),(notes,'notes')]:
            widget.delete('1.0','end'); widget.insert('1.0',row[field]); widget.edit_modified(False)
        if row['reviewer_id']: name.set(row['reviewer_id'])
        photo_id.set(f'Photo {index+1} of {len(store.rows)} — {row["image_id"]}')
        with Image.open(store.root / row['image_path']) as original:
            current_image = ImageOps.exif_transpose(original).convert('RGB')
            current_image.thumbnail((2200,2200))
        loading = False; dirty = False; update_progress(); window.after(50, render)
        status.set('Previously completed. Editing will save a draft.' if row['review_status']=='complete' else 'Choose all five answers and describe what you can see. “Not sure” is allowed.')

    def move(delta):
        nonlocal index
        if flush(): index = max(0,min(len(store.rows)-1,index+delta)); show()

    def save_next():
        nonlocal index
        if not flush(complete=True): return
        if index < len(store.rows)-1: index += 1; show()
        elif all(row['review_status']=='complete' for row in store.rows):
            messagebox.showinfo('Review complete',f'All answers are saved. Return {store.output.name}.\n\nKeep your answers private until the other reviewer finishes.',parent=window)
        else:
            index = next(i for i,row in enumerate(store.rows) if row['review_status']!='complete'); show()

    def help_window():
        guide = tk.Toplevel(window); guide.title('How to judge the photo'); guide.geometry('820x650')
        text = ScrolledText(guide, wrap='word', font=('Segoe UI',11), padx=15, pady=15); text.pack(fill='both',expand=True)
        text.insert('1.0',(store.root/'CRITERIA.md').read_text(encoding='utf-8')); text.configure(state='disabled')

    def close():
        if flush() or messagebox.askyesno('Unsaved changes','Your latest changes could not be saved. Close anyway?',parent=window): window.destroy()

    for variable in [name,*variables.values()]: variable.trace_add('write',changed)
    for widget in (evidence,notes): widget.bind('<<Modified>>',text_changed)
    photo.bind('<Configure>',resized)
    ttk.Button(buttons,text='Criteria / help',command=help_window).pack(side='left')
    ttk.Button(buttons,text='Open original photo',command=lambda: os.startfile(str(store.root/store.rows[index]['image_path']))).pack(side='left',padx=8)
    ttk.Button(buttons,text='Save & next  ·  Ctrl+Enter',style='Primary.TButton',command=save_next).pack(side='right')
    ttk.Button(buttons,text='Next (draft)',command=lambda: move(1)).pack(side='right',padx=8)
    ttk.Button(buttons,text='Previous',command=lambda: move(-1)).pack(side='right')
    window.bind('<Control-Return>',lambda _:save_next()); window.protocol('WM_DELETE_WINDOW',close)
    show(); window.mainloop()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--reviewer',choices=['reviewer1','reviewer2'],required=True)
    args = parser.parse_args(argv)
    try:
        if os.name == 'nt':
            import msvcrt
            with (args.root/f'.{args.reviewer}.lock').open('a+b') as lock:
                lock.seek(0)
                try: msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
                except OSError: raise RuntimeError('This reviewer is already open. Use the existing review window or close it first.')
                launch(ReviewStore(args.root,args.reviewer),args.reviewer)
        else:
            launch(ReviewStore(args.root,args.reviewer),args.reviewer)
    except Exception as error:
        import tkinter as tk
        from tkinter import messagebox
        window=tk.Tk(); window.withdraw(); messagebox.showerror('Cannot open photo review',str(error),parent=window); window.destroy()
        raise


if __name__=='__main__': main()
