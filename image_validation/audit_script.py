import ast

with open('training/train_corrected_baseline.py', encoding='utf-8') as f:
    src = f.read()

tree = ast.parse(src)

STAGE1_value                   = None
has_EarlyStopping              = False
has_MacroF1EarlyStopping       = False
has_criterion_val_global       = 'criterion_val = None' in src
run_val_has_criterion_param    = False
per_epoch_save_in_loop         = ('epoch_ckpt_path' in src and
                                  'torch.save(model.state_dict(), epoch_ckpt_path)' in src)
test_csv_referenced            = False

for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id == 'STAGE1_EPOCHS':
                if isinstance(node.value, ast.Constant):
                    STAGE1_value = node.value.value
    if isinstance(node, ast.ClassDef):
        if node.name == 'EarlyStopping':
            has_EarlyStopping = True
        if node.name == 'MacroF1EarlyStopping':
            has_MacroF1EarlyStopping = True
    if isinstance(node, ast.FunctionDef) and node.name == 'run_validation':
        args = [a.arg for a in node.args.args]
        run_val_has_criterion_param = 'criterion' in args
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        v = node.value.lower()
        if 'test' in v and '.csv' in v:
            test_csv_referenced = True

# Loop structure: find if epoch == STAGE1_EPOCHS block comes AFTER early_stop call
# Check ordering within train() function source lines
lines = src.splitlines()
es_call_line     = next((i for i, l in enumerate(lines) if 'early_stop(epoch' in l), None)
transition_line  = next((i for i, l in enumerate(lines) if 'if epoch == STAGE1_EPOCHS' in l), None)
stage2_lr_line   = next((i for i, l in enumerate(lines) if 'LR_STAGE2' in l and 'optimizer' in l and 'Adam' in l), None)
break_line       = next((i for i, l in enumerate(lines) if 'early_stop.early_stop' in l), None)

print('=== STATIC AUDIT RESULTS ===')
print()
print('1. STAGE1_EPOCHS')
print(f'   Value: {STAGE1_value}  [Expected: 5] -> {"PASS" if STAGE1_value == 5 else "FAIL"}')
print()
print('2. EarlyStopping class (val_loss)')
print(f'   Present: {has_EarlyStopping}  [Expected: True] -> {"PASS" if has_EarlyStopping else "FAIL"}')
print()
print('3. MacroF1EarlyStopping class')
print(f'   Present: {has_MacroF1EarlyStopping}  [Expected: False] -> {"PASS" if not has_MacroF1EarlyStopping else "FAIL"}')
print()
print('4. criterion_val global variable')
print(f'   Present: {has_criterion_val_global}  [Expected: False] -> {"PASS" if not has_criterion_val_global else "FAIL"}')
print()
print('5. run_validation() criterion parameter')
print(f'   Has param: {run_val_has_criterion_param}  [Expected: True] -> {"PASS" if run_val_has_criterion_param else "FAIL"}')
print()
print('6. Per-epoch checkpoint in main loop')
print(f'   Present: {per_epoch_save_in_loop}  [Expected: True] -> {"PASS" if per_epoch_save_in_loop else "FAIL"}')
print()
print('7. test.csv referenced in any string')
print(f'   Present: {test_csv_referenced}  [Expected: False] -> {"PASS" if not test_csv_referenced else "FAIL"}')
print()
print('8. Loop order: early_stop call BEFORE stage transition')
print(f'   early_stop call at line   : {es_call_line}')
print(f'   break on early_stop at    : {break_line}')
print(f'   stage transition at line  : {transition_line}')
print(f'   Stage2 optimizer at line  : {stage2_lr_line}')
if es_call_line and break_line and transition_line:
    order_ok = es_call_line < break_line < transition_line
    print(f'   Order: ES call < break < transition -> {"PASS" if order_ok else "FAIL"} ({order_ok})')
print()
print('9. Stage transition context (lines around transition):')
if transition_line:
    for i in range(max(0, transition_line-2), min(len(lines), transition_line+6)):
        print(f'   L{i+1:3d}: {lines[i]}')
